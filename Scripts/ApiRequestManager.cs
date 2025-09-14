/*

// ApiRequestManager.cs
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

public enum HttpMethod { GET, POST, PUT, PATCH, DELETE }

public class ApiRequest
{
    public string path;
    public HttpMethod method = HttpMethod.GET;
    public string jsonBody = null;
    public Dictionary<string, string> headers = null;
    public Dictionary<string, string> query = null;
    public Action<long, string> onSuccess;
    public Action<long, string> onError;
}

public class ApiRequestManager : MonoBehaviour
{
    [Header("Server")]
    public string baseUrl = "http://127.0.0.1:8000";

    [Header("Throughput")]
    public int maxConcurrent = 8;
    public float requestsPerSecond = 50f;

    private readonly Queue<ApiRequest> _queue = new Queue<ApiRequest>();
    private int _inFlight = 0;
    private float _tokenBucket = 0f;

    void Update()
    {
        // refill tokens
        _tokenBucket += requestsPerSecond * Time.unscaledDeltaTime;

        // dispatch while we have work, tokens, and capacity
        while (_queue.Count > 0 && _inFlight < maxConcurrent && _tokenBucket >= 1f)
        {
            _tokenBucket -= 1f;
            var req = _queue.Dequeue();
            StartCoroutine(Process(req));
        }
    }

    public void Enqueue(ApiRequest request)
    {
        _queue.Enqueue(request);
    }

    private IEnumerator Process(ApiRequest r)
    {
        _inFlight++;
        string url = BuildUrl(r);
        using (UnityWebRequest uwr = BuildUnityWebRequest(url, r))
        {
            uwr.timeout = 10;
            yield return uwr.SendWebRequest();

#if UNITY_2020_2_OR_NEWER
            bool ok = uwr.result == UnityWebRequest.Result.Success;
#else
            bool ok = !(uwr.isNetworkError || uwr.isHttpError);
#endif
            long code = uwr.responseCode;
            string body = uwr.downloadHandler != null ? uwr.downloadHandler.text : "";

            if (ok) r.onSuccess?.Invoke(code, body);
            else    r.onError?.Invoke(code, $"{uwr.error} (Allow: {uwr.GetResponseHeader("Allow")})");
        }
        _inFlight--;
    }

    private string BuildUrl(ApiRequest r)
    {
        string path = r.path ?? "";
        if (!path.StartsWith("/")) path = "/" + path;

        var sb = new StringBuilder();
        sb.Append(baseUrl.TrimEnd('/')).Append(path);

        if (r.query != null && r.query.Count > 0)
        {
            bool first = true;
            foreach (var kv in r.query)
            {
                sb.Append(first ? "?" : "&");
                first = false;
                sb.Append(UnityWebRequest.EscapeURL(kv.Key))
                  .Append("=")
                  .Append(UnityWebRequest.EscapeURL(kv.Value ?? ""));
            }
        }
        return sb.ToString();
    }

    private UnityWebRequest BuildUnityWebRequest(string url, ApiRequest r)
    {
        string verb = r.method.ToString();
        UnityWebRequest uwr;

        if (r.method == HttpMethod.GET)
        {
            uwr = UnityWebRequest.Get(url);
        }
        else
        {
            byte[] payload = string.IsNullOrEmpty(r.jsonBody) ? Array.Empty<byte>() : Encoding.UTF8.GetBytes(r.jsonBody);
            uwr = new UnityWebRequest(url, verb);
            uwr.uploadHandler = new UploadHandlerRaw(payload);
            uwr.downloadHandler = new DownloadHandlerBuffer();
            uwr.SetRequestHeader("Content-Type", "application/json");
        }

        if (r.headers != null)
            foreach (var h in r.headers)
                uwr.SetRequestHeader(h.Key, h.Value);

        return uwr;
    }
}
*/