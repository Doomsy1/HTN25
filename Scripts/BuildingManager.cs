using UnityEngine;
using System.Collections.Generic;

public class BuildingManager : MonoBehaviour{
    public static BuildingManager Instance { get; private set; }
    [SerializeField] private List<GameObject> bases;
    [SerializeField] private List<GameObject> layers;
    [SerializeField] private List<GameObject> roofs;
    [SerializeField] private List<Material> blockMaterials;

    private void Awake(){
        if (Instance != null && Instance != this){
            Destroy(gameObject); 
            return;
        }
        Instance = this;
    }

    public void RequestBuilding(GameObject tile, Vector3 position){
        bool mustBeLayer = false;
        List<GameObject> picker ;
        int buildingHeight = Random.Range(4, 14);
        float buildingHigh = 0f;
        int first = buildingHeight;

        while (buildingHeight != 0){
            if (!mustBeLayer){
                if (Random.Range(0, 1) == 0){
                    picker = bases;
                    mustBeLayer = true;
                } else{
                    picker = layers;
                    mustBeLayer = false;
                }
            } else{
                picker = layers;
                mustBeLayer = false;
            }
            if (buildingHeight == 1) picker = roofs;

            GameObject piece = picker[Random.Range(0, picker.Count)];

            if (first == buildingHeight){
                buildingHigh += piece.GetComponent<BuildingBlock>().blockHeight / 2;
                buildingHigh += 1f;
            }

            GameObject newPiece = Instantiate(piece, position + new Vector3(0, buildingHigh, 0), Quaternion.identity);
            buildingHigh += newPiece.GetComponent<BuildingBlock>().blockHeight;

            Material randomMat = blockMaterials[Random.Range(0, blockMaterials.Count)];
            foreach(Transform child in newPiece.transform){
                child.gameObject.GetComponent<MeshRenderer>().material = randomMat;
            }

            buildingHeight -= 1;
        }
        
    }
}