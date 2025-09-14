using UnityEngine;

public class KillMob : MonoBehaviour{
    public LayerMask collisionMask;

    void OnCollisionEnter(Collision collision) {
        if ((collisionMask.value & (1 << collision.gameObject.layer)) != 0) {
            collision.gameObject.GetComponent<Enemy>().enabled = false;
            collision.gameObject.GetComponent<Rigidbody>().AddForce(collision.contacts[0].normal * 10f, ForceMode.Impulse);
        }
    }

    void OnTriggerEnter(Collider other) {
    if ((collisionMask.value & (1 << other.gameObject.layer)) != 0) {
        other.GetComponent<Enemy>().enabled = false;

        Vector3 collisionPoint = other.ClosestPoint(transform.position);
        Vector3 collisionNormal = (collisionPoint - transform.position).normalized;

        Destroy(other.gameObject);
    }
}

}
