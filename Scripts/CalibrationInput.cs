using UnityEngine;
using TMPro;
public class CalibrationInput : MonoBehaviour{
    [SerializeField] private TMP_Text calTMPText;
    [SerializeField] private TMP_Text playTMPText;
    private bool calibrated = false;
    [SerializeField] private Material redMaterial;
    [SerializeField] private Material greenMaterial;
    [SerializeField] private MeshRenderer button;
    [SerializeField] private Player player;
    [SerializeField] private PlayerBall playerBallReal;
    [SerializeField] private PlayerBall playerBallMenu;
    [SerializeField] private InputHandler inputHandler;
    [SerializeField] private Calibration calibration;


    public void Awake(){
        calTMPText.text = "Calibration Required";
        playTMPText.text = "Error";
        button.material = redMaterial;
        

        inputHandler.CInputModule.preformedDelegate += GetCalibration;
        inputHandler.VInputModule.preformedDelegate += GetBallCalibration;
    }

    public void GetBallCalibration(float input){
        if (input == 0f) return;
        calibration.RequestBallCal();
    }


    public void GetCalibration(float input){
        if (input == 0f) return;
        calibration.RequestCorners();
        Vector2 firstCorner = new Vector2(0, 0);
       // calibration.


        calibrated = true;
        calTMPText.text = "";
        playTMPText.text = "Play";
        button.material = greenMaterial;
    }

    public void GetPlay(){
        Debug.Log("okbro");
        if (!calibrated) return;
        player.go = true;
        playerBallMenu.enabled = false;
        playerBallReal.enabled = true;

    }
}
