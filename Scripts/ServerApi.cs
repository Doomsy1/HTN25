
/*
// ServerApi.cs
using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable] public class CalibrateResponse { public bool started; }

// Matches get_ball JSON: { "t": <double>, "position_m": [x,y,z], "velocity_mps": [vx,vy,vz] }
[Serializable] public class BallEvent
{
    public double t;
    public float[] position_m;
    public float[] velocity_mps;
}

public class ServerApi : MonoBehaviour
{
    [Tooltip("Reference to the ApiRequestManager handling queue, rate limit and concurrency.")]
    public ApiRequestManager api;

    private void Awake()
    {
        if (!api) api = FindObjectOfType<ApiRequestManager>();
        if (!api) Debug.LogError("[ServerApi] ApiRequestManager not found. Please add one to the scene and assign it.");
    }

    // ========= /calibrate (POST) =========
    // Returns 202 Accepted with {"started": true}; 409 if already running.
    public void Calibrate(
        Action<CalibrateResponse> onAccepted,
        Action<long, string> onError = null,
        string jsonBody = null,
        Dictionary<string, string> headers = null)
    {
        api.Enqueue(new ApiRequest
        {
            path = "/calibrate",
            method = HttpMethod.POST,
            jsonBody = jsonBody ?? "{}",   // server ignores body, but keep valid JSON
            headers = headers,
            onSuccess = (code, body) =>
            {
                try
                {
                    // FastAPI returns a small object; guard for empty body just in case
                    var parsed = string.IsNullOrEmpty(body)
                        ? new CalibrateResponse { started = true }
                        : JsonUtility.FromJson<CalibrateResponse>(body);
                    onAccepted?.Invoke(parsed);
                }
                catch (Exception ex)
                {
                    onError?.Invoke(code, $"Parse error: {ex.Message}; body: {body}");
                }
            },
            onError = (code, err) =>
            {
                // Common case: 409 when calibration already in progress
                onError?.Invoke(code, err);
            }
        });
    }

    // ========= /is_calibrated (GET) =========
    // Returns plain JSON true/false (not an object).
    public void IsCalibrated(Action<bool> onSuccess, Action<long, string> onError = null)
    {
        api.Enqueue(new ApiRequest
        {
            path = "/is_calibrated",
            method = HttpMethod.GET,
            onSuccess = (code, body) =>
            {
                var s = (body ?? "").Trim().ToLowerInvariant();
                if (s == "true")  onSuccess?.Invoke(true);
                else if (s == "false") onSuccess?.Invoke(false);
                else onError?.Invoke(code, $"Unexpected /is_calibrated payload: '{body}'");
            },
            onError = (code, err) => onError?.Invoke(code, err)
        });
    }

    // ========= /get_corners (GET) =========
    // Unknown shape → offer raw string + generic typed version.
    public void GetCornersRaw(Action<string> onSuccess, Action<long, string> onError = null)
    {
        api.Enqueue(new ApiRequest
        {
            path = "/get_corners",
            method = HttpMethod.GET,
            onSuccess = (code, body) => onSuccess?.Invoke(body),
            onError = (code, err) => onError?.Invoke(code, err) // 404 until calibration exists
        });
    }

    public void GetCorners<T>(Action<T> onSuccess, Action<long, string> onError = null)
    {
        GetCornersRaw(
            body =>
            {
                try { onSuccess?.Invoke(JsonUtility.FromJson<T>(body)); }
                catch (Exception ex) { onError?.Invoke(0, $"Parse error: {ex.Message}"); }
            },
            onError
        );
    }

    // ========= /get_ball (GET) =========
    // 204 when no sample; otherwise a BallEvent object.
    public void GetBall(Action<BallEvent> onEvent, Action onNoContent = null, Action<long, string> onError = null)
    {
        api.Enqueue(new ApiRequest
        {
            path = "/get_ball",
            method = HttpMethod.GET,
            onSuccess = (code, body) =>
            {
                if (code == 204 || string.IsNullOrEmpty(body))
                {
                    onNoContent?.Invoke();
                    return;
                }
                try
                {
                    var ev = JsonUtility.FromJson<BallEvent>(body);
                    onEvent?.Invoke(ev);
                }
                catch (Exception ex)
                {
                    onError?.Invoke(code, $"Parse error: {ex.Message}; body: {body}");
                }
            },
            onError = (code, err) => onError?.Invoke(code, err)
        });
    }
}
*/