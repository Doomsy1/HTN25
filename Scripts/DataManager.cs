using UnityEngine;

public class DataManager : MonoBehaviour{
    public static DataManager Instance { get; private set; }
    [SerializeField] public Transform playerTransform;

    private void Awake(){
        if (Instance != null && Instance != this){
            Destroy(gameObject); 
            return;
        }
        Instance = this;
    }

    
}
