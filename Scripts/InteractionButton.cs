using UnityEngine;

public class InteractionButton : MonoBehaviour{
    [SerializeField] private LayerMask buttonMask;
    
    void OnTriggerEnter(Collider other) {
        Debug.Log("trigger");
        if ((buttonMask.value & (1 << other.gameObject.layer)) != 0) {
            other.gameObject.GetComponent<CalibrationInput>().GetPlay();
        }
    }
    
    void OnCollisionEnter(Collision collision) {
        Debug.Log("collide");
        if ((buttonMask.value & (1 << collision.gameObject.layer)) != 0) {
            collision.gameObject.GetComponent<CalibrationInput>().GetPlay();
        }
    }
}
