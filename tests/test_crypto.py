from dendriswarm.core.crypto import Identity, node_id_from_public_key, verify


def test_identity_signature_and_derived_id():
    identity = Identity.generate()
    value = {"hello": "world", "n": 1}
    signature = identity.sign(value)
    assert verify(identity.public_key_b64, value, signature)
    assert not verify(identity.public_key_b64, {"hello": "tampered"}, signature)
    assert identity.node_id == node_id_from_public_key(identity.public_key_b64)
