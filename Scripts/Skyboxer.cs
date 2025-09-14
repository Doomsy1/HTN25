using UnityEngine;

public class Skyboxer : MonoBehaviour{
    [SerializeField] private Material skyboxMat;
    [SerializeField] private float blend;

    public void Update(){
        skyboxMat.SetFloat("_Blend", blend);
    }
}
