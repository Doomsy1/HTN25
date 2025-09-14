using UnityEngine;

public class Enemy : MonoBehaviour{
    [SerializeField] private Rigidbody rb;
    [SerializeField] private float speed;
    public bool aggro = false;

    void Update(){
        if (aggro){
            rb.MovePosition(((DataManager.Instance.playerTransform.position - transform.position).normalized * (1/speed)) + transform.position);
            transform.LookAt(DataManager.Instance.playerTransform);
        }
    }
}
