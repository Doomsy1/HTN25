using UnityEngine;
using System;

#nullable enable
public class InputModule<T>{
    public event Action<T>? preformedDelegate; 
    public event Action<T>? startedDelegate; 
    public event Action? canceledDelegate; 

    public void TriggerPreformed(T value){preformedDelegate?.Invoke(value);}
    public void TriggerStarted(T value){startedDelegate?.Invoke(value);}
    public void TriggerCanceled(){canceledDelegate?.Invoke();}
}
#nullable disable

public class InputHandler: MonoBehaviour{
    private InputSystem inputSystem;
    private InputSystem.PlayerActions playerMovement;

    public InputModule<Vector2> mouseInputModule = new InputModule<Vector2>();
    public InputModule<Vector2> keyboardInputModule = new InputModule<Vector2>();
    public InputModule<float> LMBInputModule = new InputModule<float>();
    public InputModule<Vector2> mousePosInputModule = new InputModule<Vector2>();
    public InputModule<float> CInputModule = new InputModule<float>();
    public InputModule<float> VInputModule = new InputModule<float>();

    public void Awake(){
        inputSystem = new InputSystem();
        playerMovement = inputSystem.Player;
        
        playerMovement.MOUSE.performed += ctx => mouseInputModule.TriggerPreformed(ctx.ReadValue<Vector2>());
        playerMovement.MOUSE.started += ctx => mouseInputModule.TriggerStarted(ctx.ReadValue<Vector2>());
        playerMovement.MOUSE.canceled += ctx => mouseInputModule.TriggerCanceled();

        playerMovement.KEYBOARD.performed += ctx => keyboardInputModule.TriggerPreformed(ctx.ReadValue<Vector2>());
        playerMovement.KEYBOARD.started += ctx => keyboardInputModule.TriggerStarted(ctx.ReadValue<Vector2>());
        playerMovement.KEYBOARD.canceled += ctx => keyboardInputModule.TriggerCanceled();

        playerMovement.LMB.performed += ctx => LMBInputModule.TriggerPreformed(ctx.ReadValue<float>());
        playerMovement.LMB.started += ctx => LMBInputModule.TriggerStarted(ctx.ReadValue<float>());
        playerMovement.LMB.canceled += ctx => LMBInputModule.TriggerCanceled();

        playerMovement.MOUSEPOS.performed += ctx => mousePosInputModule.TriggerPreformed(ctx.ReadValue<Vector2>());
        playerMovement.MOUSEPOS.started += ctx => mousePosInputModule.TriggerStarted(ctx.ReadValue<Vector2>());
        playerMovement.MOUSEPOS.canceled += ctx => mousePosInputModule.TriggerCanceled();

        playerMovement.C.performed += ctx => CInputModule.TriggerPreformed(ctx.ReadValue<float>());
        playerMovement.C.started += ctx => CInputModule.TriggerStarted(ctx.ReadValue<float>());
        playerMovement.C.canceled += ctx => CInputModule.TriggerCanceled();

        playerMovement.V.performed += ctx => VInputModule.TriggerPreformed(ctx.ReadValue<float>());
        playerMovement.V.started += ctx => VInputModule.TriggerStarted(ctx.ReadValue<float>());
        playerMovement.V.canceled += ctx => VInputModule.TriggerCanceled();

    }

    private void OnEnable() {playerMovement.Enable();}
    private void OnDisable() {playerMovement.Disable();}

    public void Start(){
        //Cursor.lockState = CursorLockMode.Locked;
        //Cursor.visible = false;
    }
}
