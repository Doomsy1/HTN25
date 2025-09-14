using UnityEngine;

public class PlayerCamera : MonoBehaviour{
    [SerializeField] private Transform fullRot;
    [SerializeField] private InputHandler inputHandler;
    [SerializeField] private float sensX, sensY;

    private float fullRotAngle = 0f;

    public void Awake(){
        inputHandler.mouseInputModule.preformedDelegate += RecieveMouseInput;
    }

    public void RecieveMouseInput(Vector2 mouseInput){
        transform.rotation *= Quaternion.Euler(0f, mouseInput.x * Time.deltaTime * sensX, 0f);
        fullRotAngle += mouseInput.y * -sensY * Time.deltaTime;
        fullRotAngle = Mathf.Clamp(fullRotAngle, -90, 90);
        fullRot.rotation = transform.rotation * Quaternion.Euler(fullRotAngle, 0f, 0f);
    }
}
 