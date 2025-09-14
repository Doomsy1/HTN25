using UnityEngine;
using System.Collections.Generic;

public class BuildingSpots : MonoBehaviour{
    [SerializeField] private List<Transform> buildingPositions;

    public void Awake(){
        foreach (Transform t in buildingPositions){
            BuildingManager.Instance.RequestBuilding(gameObject, t.position);
        }
    }
    
}
