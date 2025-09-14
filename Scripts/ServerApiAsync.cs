// ServerApiAsync.cs
// One-method-per-endpoint with async/await return values.
// Requires .NET 4.x scripting runtime. Works in Editor/Standalone/WebGL.

using System;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

[Serializable] public class CalibrateResponse { public bool started; }

// { "t": <double>, "position_m": [x,y,z], "velocity_mps": [vx,vy,vz] }
[Serializable] public class BallEvent
{
    public double t;
    public float[] position_m;
    public float[] velocity_mps;
}

// Optional: throw this on non-success HTTP codes (except where handled, like 204 on /get_ball)
public class ApiException : Exception
{
    public long StatusCode { get; }
    public string Body { get; }
    public ApiException(long status, string body, string message) : base(message)
    {
        StatusCode = status; Body = body;
    }
}

public class ServerApiAsync : MonoBehaviour
{
    [Header("Server")]
    [Tooltip("Base URL, e.g., http://127.0.0.1:8000")]
    public string baseUrl = "http://127.0.0.1:8000";

    [Tooltip("UnityWebRequest timeout in seconds")]
    public int timeoutSeconds = 10;

    // ---------- Public API: one function per endpoint ----------

    /// POST /calibrate  => 202 {"started": true}  or 409 if already running
    public async Task<CalibrateResponse> CalibrateProjectorAsync(string jsonBody = "{}")
    {
        var (code, body) = await SendAsync("POST", "/calibrate_projector", jsonBody, "application/json");
        // FastAPI returns a tiny JSON object on 202; if empty, assume started
        if (string.IsNullOrEmpty(body)) return new CalibrateResponse { started = code == 202 };
        try { return JsonUtility.FromJson<CalibrateResponse>(body); }
        catch (Exception ex) { throw new ApiException(code, body, $"Parse error: {ex.Message}"); }
    }

    public async Task<CalibrateResponse> CalibrateBallAsync(string jsonBody = "{}")
    {
        var (code, body) = await SendAsync("POST", "/calibrate_ball", jsonBody, "application/json");
        // FastAPI returns a tiny JSON object on 202; if empty, assume started
        if (string.IsNullOrEmpty(body)) return new CalibrateResponse { started = code == 202 };
        try { return JsonUtility.FromJson<CalibrateResponse>(body); }
        catch (Exception ex) { throw new ApiException(code, body, $"Parse error: {ex.Message}"); }
    }

    /// GET /is_calibrated  => "true" or "false"
    public async Task<bool> IsCalibratedAsync()
    {
        var (code, body) = await SendAsync("GET", "/is_calibrated");
        var s = (body ?? "").Trim().ToLowerInvariant();
        if (s == "true") return true;
        if (s == "false") return false;
        throw new ApiException(code, body, $"Unexpected payload for /is_calibrated: '{body}'");
    }

    /// GET /get_corners  => arbitrary JSON (404 until calibration exists)
    public async Task<string> GetCornersRawAsync()
    {
        var (_, body) = await SendAsync("GET", "/get_corners");
        return body;
    }

    /// Strongly-typed version for /get_corners if you know the schema
    public async Task<T> GetCornersAsync<T>()
    {
        var (_, body) = await SendAsync("GET", "/get_corners");
        try { return JsonUtility.FromJson<T>(body); }
        catch (Exception ex) { throw new ApiException(0, body, $"Parse error: {ex.Message}"); }
    }

    /// GET /get_ball  => 204 No Content (return null) or BallEvent
    public async Task<BallEvent> GetBallAsync()
    {
        var (code, body) = await SendAsync("GET", "/get_ball", treat204AsSuccess: true);
        if (code == 204 || string.IsNullOrEmpty(body)) return null; // no sample available
        try { return JsonUtility.FromJson<BallEvent>(body); }
        catch (Exception ex) { throw new ApiException(code, body, $"Parse error: {ex.Message}"); }
    }

    // ---------- Internals ----------

    private async Task<(long status, string body)> SendAsync(
        string method,
        string path,
        string body = null,
        string contentType = null,
        bool treat204AsSuccess = false)
    {
        string url = $"{baseUrl.TrimEnd('/')}{(path.StartsWith("/") ? path : "/" + path)}";

        using (var req = BuildRequest(url, method, body, contentType))
        {
            req.timeout = Mathf.Max(1, timeoutSeconds);

            // Await the UnityWebRequest like a Task
            var op = req.SendWebRequest();
            var tcs = new TaskCompletionSource<bool>();
            op.completed += _ => tcs.TrySetResult(true);
            await tcs.Task;

            long code = req.responseCode;
#if UNITY_2020_2_OR_NEWER
            bool ok = req.result == UnityWebRequest.Result.Success;
#else
            bool ok = !(req.isNetworkError || req.isHttpError);
#endif
            string resp = req.downloadHandler != null ? req.downloadHandler.text : "";

            // Consider 204 a "success" when requested (used by /get_ball)
            if (treat204AsSuccess && code == 204) return (code, "");

            if (!ok)
                throw new ApiException(code, resp, $"HTTP {code} — {req.error}");

            return (code, resp);
        }
    }

    private UnityWebRequest BuildRequest(string url, string method, string body, string contentType)
    {
        if (string.Equals(method, "GET", StringComparison.OrdinalIgnoreCase))
        {
            var get = UnityWebRequest.Get(url);
            get.downloadHandler = new DownloadHandlerBuffer();
            return get;
        }

        var uwr = new UnityWebRequest(url, method);
        byte[] bytes = body == null ? Array.Empty<byte>() : Encoding.UTF8.GetBytes(body);
        uwr.uploadHandler = new UploadHandlerRaw(bytes);
        uwr.downloadHandler = new DownloadHandlerBuffer();
        if (!string.IsNullOrEmpty(contentType))
            uwr.SetRequestHeader("Content-Type", contentType);
        return uwr;
    }
}
