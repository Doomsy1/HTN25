using UnityEngine;
using System.Collections.Generic;

public class TileManager : MonoBehaviour{
    [SerializeField] private TileFolder tileFolder;
    [SerializeField] private float bufferDistance; 
    [SerializeField] private Transform player;
    [SerializeField] private int addTileBuffer;
    [SerializeField] private float tileSize;

    private TileCollection currentTileCollection;
    private int tileBuffer; 
    private Vector3 nextTilePosition;
    private List<GameObject> tiles;

    public void Awake(){
        tileFolder.Initialize();
        tiles = new List<GameObject>();
        SelectNewTileCollection();
        tileBuffer = (int) Mathf.Ceil(bufferDistance/tileSize) + addTileBuffer;
    }

    public void SelectNewTileCollection(){
        currentTileCollection = tileFolder.tileFolder[Random.Range(0, tileFolder.length)];
    }

    public void SpawnNextTile(Vector3 playerPosition){
        if (playerPosition.x + bufferDistance > nextTilePosition.x){
            GameObject tilePrefab = currentTileCollection.tileCollection[Random.Range(0, currentTileCollection.length)].tilePrefab;
            if (tilePrefab == null){ 
                SelectNewTileCollection();
                SpawnNextTile(playerPosition);
                return;
            } 
            GameObject newTile = Instantiate(tilePrefab, nextTilePosition, Quaternion.identity);
            nextTilePosition += new Vector3(tileSize, 0, 0);
            if (tiles.Count == tileBuffer){
                GameObject deleteTile = tiles[0];
                tiles.RemoveAt(0);
                Destroy(deleteTile);
            } 
            tiles.Add(newTile);
        }
    }
    
    public void RollTile(){
        
    }

    public void Update(){
        SpawnNextTile(player.position);
    }


}
