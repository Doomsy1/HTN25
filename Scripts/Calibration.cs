

using UnityEngine;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Globalization;
using System.Collections.Concurrent;

public class Calibration : MonoBehaviour
{
    [SerializeField] private PlayerBall playerBall;
    [SerializeField] private InputHandler inputHandler;
    [SerializeField] private Camera cam;
    [SerializeField] private LayerMask ignoreLayerMask;
    [SerializeField] private ServerApiAsync api;

    [Header("Polling")]
    [SerializeField] private float pollHz = 60f;
    [SerializeField] private bool startPollingOnAwake = true;
    [SerializeField] private bool logApiErrors = true;

    private float scaleX;
    private float scaleY;
    private Vector2 screenSize = new Vector2(1920, 1080);
    private readonly List<Vector2> fourCorners = new List<Vector2>();

    // async polling infra
    private CancellationTokenSource _cts;
    private SynchronizationContext _mainCtx;
    private readonly ConcurrentQueue<System.Action> _mainThreadQueue = new ConcurrentQueue<System.Action>();

    public void Awake(){
        _mainCtx = SynchronizationContext.Current;

        if (inputHandler != null && inputHandler.mousePosInputModule != null)
            inputHandler.mousePosInputModule.preformedDelegate += RecieveMousePosInput;
        
        //TempRequest(); // temp scaling until RequestCorners runs

        if (startPollingOnAwake)
            StartBallPolling();
    }


    private void OnEnable()
    {
        if (_cts == null && startPollingOnAwake) StartBallPolling();
    }

    private void OnDisable() => StopBallPolling();
    private void OnDestroy() => StopBallPolling();

    private void Update()
    {
        // Fallback dispatcher in case SyncContext.Post isn't available
        while (_mainThreadQueue.TryDequeue(out var a))
        {
            try { a(); } catch (System.Exception ex) { Debug.LogException(ex); }
        }
    }

    public async void RequestBallCal(){
        try{
            var calB = await api.CalibrateBallAsync();
            Debug.Log($"Calibrate Ball started: {calB.started}");

            while (!await api.IsCalibratedAsync())
                await Task.Delay(200);

            Debug.Log("Calibrated ✅");

    
        } catch (ApiException ex) {
            Debug.LogError($"API error {ex.StatusCode}: {ex.Message}\nBody: {ex.Body}");
        }
    }

    // ---------- Corner calibration flow (unchanged, just as you had) ----------
    public async void RequestCorners(){
        try{
            var calP = await api.CalibrateProjectorAsync();
            Debug.Log($"Calibrate Projector started: {calP.started}");

            while (!await api.IsCalibratedAsync())
                await Task.Delay(200);

            Debug.Log("Calibrated ✅");

            var cornersJson = await api.GetCornersRawAsync();
            Debug.Log($"Corners JSON: {cornersJson}");

            int afterIndex = cornersJson.IndexOf("screen_corners_3d_m");
            string[] parsed = cornersJson.Substring(afterIndex+24).Replace(" ", "").Replace("[", "").Replace("]", "").Split(",");

            Vector2 corner1 = new Vector2(float.Parse(parsed[0], CultureInfo.InvariantCulture), float.Parse(parsed[1], CultureInfo.InvariantCulture));
            Vector2 corner2 = new Vector2(float.Parse(parsed[3], CultureInfo.InvariantCulture), float.Parse(parsed[4], CultureInfo.InvariantCulture));
            Vector2 corner3 = new Vector2(float.Parse(parsed[6], CultureInfo.InvariantCulture), float.Parse(parsed[7], CultureInfo.InvariantCulture));
            Vector2 corner4 = new Vector2(float.Parse(parsed[9], CultureInfo.InvariantCulture), float.Parse(parsed[10], CultureInfo.InvariantCulture));

            fourCorners.Clear();
            fourCorners.Add(corner1);
            fourCorners.Add(corner2);
            fourCorners.Add(corner3);
            fourCorners.Add(corner4);
        } catch (ApiException ex) {
            Debug.LogError($"API error {ex.StatusCode}: {ex.Message}\nBody: {ex.Body}");
        }

        if (fourCorners.Count >= 3){
            scaleX = screenSize.x / (fourCorners[2].x - fourCorners[0].x);
            scaleY = screenSize.y / Mathf.Abs(fourCorners[2].y - fourCorners[0].y);
        }

        Debug.Log(scaleX + " scaleX");
        Debug.Log(scaleY + " scaleY");
    }

    public void TempRequest(){
        Vector2 corner3 = new Vector2(1920, 1080);
        Vector2 corner2 = new Vector2(0, 1080);
        Vector2 corner1 = new Vector2(0, 0);
        Vector2 corner4 = new Vector2(1920, 0);

        fourCorners.Clear();
        fourCorners.Add(corner1);
        fourCorners.Add(corner2);
        fourCorners.Add(corner3);
        fourCorners.Add(corner4);

        scaleX = screenSize.x / (fourCorners[2].x - fourCorners[0].x);
        scaleY = screenSize.y / Mathf.Abs(fourCorners[2].y - fourCorners[0].y);

        Debug.Log(scaleX);
        Debug.Log(scaleY);
    }

    // ---------- Mouse path (your existing hook) ----------
    public void RecieveMousePosInput(Vector2 screenPosition){
        Input(screenPosition);
    }

    public void Input(Vector2 realWorldPosition){
        // Map 2D "real-world" (your units) to screen pixels
        Vector2 screenPosition = new Vector2(realWorldPosition.x * scaleX, realWorldPosition.y * scaleY);

        Vector3 targetPosition;
        Ray ray = cam.ScreenPointToRay(screenPosition);
        if (Physics.Raycast(ray, out RaycastHit hit, 0.01f, ~ignoreLayerMask)){
            targetPosition = hit.point;
        }
        else{
            targetPosition = ray.origin + ray.direction - cam.transform.position;
        }

        playerBall.SimulateBall(targetPosition);
    }

    // ---------- New API-driven path (3D pos/vel) ----------
    // Called by the poller whenever /get_ball yields a sample.
    public void Input(Vector3 realWorldPosition, Vector3 velocity){
        // Convert 3D “meters” into your existing 2D mapping using x/y.
        // If your mapping should incorporate Z, adjust here.
        Input(new Vector2(realWorldPosition.x, realWorldPosition.y));
        // If PlayerBall supports velocity, pass it here as well.
        // e.g., playerBall.SimulateBallWithVelocity(targetPosition, velocity);
    }

    // ---------- Async poller embedded here ----------
    public void StartBallPolling()
    {
        if (_cts != null) return;
        _cts = new CancellationTokenSource();
        _ = PollLoopAsync(_cts.Token);
    }

    public void StopBallPolling()
    {
        _cts?.Cancel();
        _cts?.Dispose();
        _cts = null;
    }

    private async Task PollLoopAsync(CancellationToken token)
    {
        if (api == null)
        {
            Debug.LogError("[Calibration] ServerApiAsync reference is missing.");
            return;
        }

        int intervalMs = Mathf.Max(1, Mathf.RoundToInt(1000f / Mathf.Max(1f, pollHz)));

        while (!token.IsCancellationRequested)
        {
            try
            {
                Debug.Log("trying to get ball");
                var ev = await api.GetBallAsync(); // null when 204
                if (ev != null && ev.position_m != null && ev.position_m.Length >= 3 &&
                    ev.velocity_mps != null && ev.velocity_mps.Length >= 3)
                {
                    var pos = new Vector3(ev.position_m[0], ev.position_m[1], ev.position_m[2]);
                    var vel = new Vector3(ev.velocity_mps[0], ev.velocity_mps[1], ev.velocity_mps[2]);
                    Debug.Log("got a ball");

                    // Ensure we run Input(...) on Unity main thread
                    DispatchToMainThread(() => Input(pos, vel));
                }
            }
            catch (ApiException ex)
            {
                if (logApiErrors) Debug.LogWarning($"[Calibration] API error {ex.StatusCode}: {ex.Message}");
            }
            catch (System.Exception ex)
            {
                if (logApiErrors) Debug.LogException(ex);
            }

            try { await Task.Delay(intervalMs, token); } catch { /* cancelled */ }
        }
    }

    private void DispatchToMainThread(System.Action action)
    {
        if (action == null) return;

        if (_mainCtx != null)
        {
            try { _mainCtx.Post(_ => action(), null); return; }
            catch { /* fall through to queue */ }
        }
        _mainThreadQueue.Enqueue(action);
    }
}


/*

using UnityEngine;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Globalization;

public class Calibration : MonoBehaviour{
    [SerializeField] private PlayerBall playerBall;
    [SerializeField] private InputHandler inputHandler;
    [SerializeField] private Camera cam;
    [SerializeField] private LayerMask ignoreLayerMask;
    [SerializeField] private ServerApiAsync api;

    private float scaleX;
    private float scaleY;
    private Vector2 screenSize = new Vector2(1920, 1080);
    private List<Vector2> fourCorners = new List<Vector2>();

    public void Awake(){
        inputHandler.mousePosInputModule.preformedDelegate += RecieveMousePosInput;
        TempRequest();
    }

    public async void RequestCorners(){
        try{
            var cal = await api.CalibrateAsync();
            Debug.Log($"Calibrate started: {cal.started}");

            while (!await api.IsCalibratedAsync())
                await Task.Delay(200);

            Debug.Log("Calibrated ✅");

            var cornersJson = await api.GetCornersRawAsync();
            Debug.Log($"Corners JSON: {cornersJson}");

            int afterIndex = cornersJson.IndexOf("screen_corners_3d_m");
            string[] parsed = cornersJson.Substring(afterIndex+24).Replace(" ", "").Replace("[", "").Replace("]", "").Split(",");

            foreach(string part in parsed){
                Debug.Log("part " + part);
            }

            Vector2 corner1 = new Vector2(float.Parse(parsed[0], CultureInfo.InvariantCulture), float.Parse(parsed[1], CultureInfo.InvariantCulture));
            Vector2 corner2 = new Vector2(float.Parse(parsed[3], CultureInfo.InvariantCulture), float.Parse(parsed[4], CultureInfo.InvariantCulture));
            Vector2 corner3 = new Vector2(float.Parse(parsed[6], CultureInfo.InvariantCulture), float.Parse(parsed[7], CultureInfo.InvariantCulture));
            Vector2 corner4 = new Vector2(float.Parse(parsed[9], CultureInfo.InvariantCulture), float.Parse(parsed[10], CultureInfo.InvariantCulture));

            fourCorners.Add(corner1);
            fourCorners.Add(corner2);
            fourCorners.Add(corner3);
            fourCorners.Add(corner4);
        } catch (ApiException ex) {
            Debug.LogError($"API error {ex.StatusCode}: {ex.Message}\nBody: {ex.Body}");
        }
        
        scaleX = screenSize.x / (fourCorners[2].x - fourCorners[0].x);
        scaleY = screenSize.y / Mathf.Abs(fourCorners[2].y - fourCorners[0].y);
    }

    public void TempRequest(){
        Vector2 corner3 = new Vector2(1920, 1080);
        Vector2 corner2 = new Vector2(0, 1080);
        Vector2 corner1 = new Vector2(0, 0);
        Vector2 corner4 = new Vector2(1920, 0);

        fourCorners.Add(corner1);
        fourCorners.Add(corner2);
        fourCorners.Add(corner3);
        fourCorners.Add(corner4);

        scaleX = screenSize.x / (fourCorners[2].x - fourCorners[0].x);
        scaleY = screenSize.y / Mathf.Abs(fourCorners[2].y - fourCorners[0].y);

        Debug.Log(scaleX);
        Debug.Log(scaleY);
    }

    public void RecieveMousePosInput(Vector2 screenPosition){
        Input(screenPosition);
    }

    public void Input(Vector2 realWorldPosition){
        //Vector2 screenPosition = new Vector2((realWorldPosition.x - fourCorners[0].x) * scaleX,  1080 - (fourCorners[0].y - realWorldPosition.y) * scaleY);
        Vector2 screenPosition = new Vector2(realWorldPosition.x * scaleX, realWorldPosition.y * scaleY);
        
        Vector3 targetPosition;

        Ray ray = cam.ScreenPointToRay(screenPosition);
        if (Physics.Raycast(ray, out RaycastHit hit, 0.01f, ~ignoreLayerMask)){
            targetPosition = hit.point;
        }
        else{
            Debug.Log(ray.origin);
            targetPosition = ray.origin + ray.direction - cam.transform.position;
        }

        playerBall.SimulateBall(targetPosition);
    }
}

*/