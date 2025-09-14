using UnityEngine;

public class Player : MonoBehaviour{
    [SerializeField] private Rigidbody rb;
    public bool go = false;

    public void Update(){
        if (!go) return;
        rb.linearVelocity = new Vector3(20f, 0, 0);
    }
}
