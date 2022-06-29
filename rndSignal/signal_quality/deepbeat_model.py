
# Load the trained model
import logging
import os
import tensorflow.compat.v1 as tf
# suppress the error messages
tf.disable_v2_behavior()
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.autograph.set_verbosity(3)

def _load_model():
    """Miscellaneous function to load the deepbeat model.
    
    Returns 
    ------
    deepbeat: Tensorflow model
    
    """
    #print("Loading DeepBeat trained model")
    # will change the directory if necessary
    base_path = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_path, "deepbeat.h5")
    deepbeat = tf.keras.models.load_model(filepath)
    
    #print("Done!")
    
    return deepbeat

if __name__ == "__main__":
    load_model()
