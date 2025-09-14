using UnityEngine;

public class PlayerBall : MonoBehaviour{
    [SerializeField] private GameObject ballPrefab;
    [SerializeField] private Transform camTransform;
    [SerializeField] private float range;
    [SerializeField] private float strength;

    public void Awake(){
    }
    
    public Vector3 RandomVelocity(){
        return new Vector3(Random.Range(-range, range),Random.Range(-range, range),Random.Range(-range, range));
    }

    public void SimulateBall(Vector3 velocity){
        GameObject newBall = Instantiate(ballPrefab, camTransform.position + new Vector3(2, 0, 0), Quaternion.identity);
        newBall.GetComponent<Rigidbody>().AddForce(velocity * strength, ForceMode.Impulse);
    }


}
