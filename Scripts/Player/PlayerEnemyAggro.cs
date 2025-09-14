using UnityEngine;

public class PlayerEnemyAggro : MonoBehaviour{
    public void OnTriggerEnter(Collider other){
        try{
            other.gameObject.GetComponent<Enemy>().aggro = true;
        } catch {
            
        }
    }
}
