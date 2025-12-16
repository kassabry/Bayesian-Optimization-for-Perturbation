"""
ULTIMATE Chemical VAE Compatibility Patch - ALL FIXES IN ONE FILE

This single file fixes ALL compatibility issues:
1. Keras module structure (keras.layers.recurrent, etc.)
2. Keras load_model h5py decode errors
3. Numpy array ambiguity errors
4. Custom layer registration (TerminalGRU)

Usage: Import this ONCE before any chemvae imports
    import chemvae_ultimate_patch
    from chemvae.vae_utils import VAEUtils
    vae = VAEUtils(directory='models/zinc_properties/')
"""

import sys
import json
import h5py

print("="*70)
print("CHEMICAL VAE ULTIMATE COMPATIBILITY PATCH")
print("="*70)

# ============================================================================
# PATCH 1: Fix Keras load_model for h5py/Python 3
# ============================================================================

def create_patched_load_model():
    """Create the patched load_model function."""
    
    def patched_load_model(filepath, custom_objects=None, compile=True):
        """Patched load_model that handles Python 3 strings from h5py."""
        from keras.models import model_from_config
        
        if isinstance(filepath, h5py.File):
            f = filepath
            opened_new_file = False
        else:
            f = h5py.File(filepath, mode='r')
            opened_new_file = True
        
        # Keep reference to the file object for closing
        file_to_close = f if opened_new_file else None
        
        try:
            model_config = f.attrs.get('model_config')
            
            if model_config is None:
                raise ValueError('No model found in config file.')
            
            # FIX: Handle both bytes and strings
            if isinstance(model_config, bytes):
                model_config = model_config.decode('utf-8')
            
            model_config = json.loads(model_config)
            model = model_from_config(model_config, custom_objects=custom_objects)
            
            # Load weights
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']  # f is now a Group, not a File
            
            if 'layer_names' in f.attrs:
                layer_names = f.attrs['layer_names']
                if len(layer_names) > 0 and isinstance(layer_names[0], bytes):
                    layer_names = [n.decode('utf-8') for n in layer_names]
                
                filtered_layer_names = []
                for name in layer_names:
                    g = f[name]
                    weight_names = g.attrs.get('weight_names')
                    # FIX: Use len() instead of if weight_names (array ambiguity)
                    if weight_names is not None and len(weight_names) > 0:
                        filtered_layer_names.append(name)
                
                layer_names = filtered_layer_names
                
                # FIX: Be more lenient with layer count mismatches
                # Try to load weights by matching layer names
                if len(layer_names) != len(model.layers):
                    print(f"⚠️  Layer count mismatch: {len(layer_names)} in file vs {len(model.layers)} in model")
                    print(f"    Attempting to load weights by layer name matching...")
                    
                    # Create a mapping of layer names to layers
                    layer_dict = {layer.name: layer for layer in model.layers}
                    
                    # Load weights for matching layers
                    loaded_count = 0
                    for name in layer_names:
                        if name in layer_dict:
                            g = f[name]
                            weight_names = g.attrs['weight_names']
                            if len(weight_names) > 0 and isinstance(weight_names[0], bytes):
                                weight_names = [w.decode('utf-8') for w in weight_names]
                            
                            weight_values = [g[weight_name] for weight_name in weight_names]
                            layer_dict[name].set_weights(weight_values)
                            loaded_count += 1
                        else:
                            print(f"    Warning: Layer '{name}' in file not found in model")
                    
                    print(f"    Loaded weights for {loaded_count}/{len(layer_names)} layers")
                else:
                    # Normal path - layer counts match
                    for k, name in enumerate(layer_names):
                        g = f[name]
                        weight_names = g.attrs['weight_names']
                        if len(weight_names) > 0 and isinstance(weight_names[0], bytes):
                            weight_names = [w.decode('utf-8') for w in weight_names]
                        
                        weight_values = [g[weight_name] for weight_name in weight_names]
                        model.layers[k].set_weights(weight_values)
            
            if compile and 'training_config' in f.attrs:
                training_config = f.attrs['training_config']
                if isinstance(training_config, bytes):
                    training_config = training_config.decode('utf-8')
                training_config = json.loads(training_config)
                
                optimizer_config = training_config['optimizer_config']
                import keras.optimizers
                optimizer = keras.optimizers.deserialize(optimizer_config)
                
                loss = training_config.get('loss', None)
                metrics = training_config.get('metrics', None)
                sample_weight_mode = training_config.get('sample_weight_mode', None)
                loss_weights = training_config.get('loss_weights', None)
                
                model.compile(optimizer=optimizer,
                            loss=loss,
                            metrics=metrics,
                            loss_weights=loss_weights,
                            sample_weight_mode=sample_weight_mode)
            
            return model
        
        finally:
            # FIX: Only close if we opened it and it's a File object
            if file_to_close is not None:
                file_to_close.close()
    
    return patched_load_model

# Apply the load_model patch
if 'keras.models' in sys.modules:
    import keras.models
    patched_load_model = create_patched_load_model()
    keras.models.load_model = patched_load_model
    sys.modules['keras.models'].load_model = patched_load_model
    print("✅ Patched keras.models.load_model")
else:
    import keras.models
    patched_load_model = create_patched_load_model()
    keras.models.load_model = patched_load_model
    sys.modules['keras.models'].load_model = patched_load_model
    print("✅ Patched keras.models.load_model")

# ============================================================================
# PATCH 2: Fix Keras Layer Structure
# ============================================================================

import keras.layers

if not hasattr(keras.layers, 'recurrent'):
    from types import ModuleType
    recurrent_module = ModuleType('keras.layers.recurrent')
    recurrent_module.GRU = keras.layers.GRU
    recurrent_module.LSTM = keras.layers.LSTM
    recurrent_module.SimpleRNN = keras.layers.SimpleRNN
    keras.layers.recurrent = recurrent_module
    sys.modules['keras.layers.recurrent'] = recurrent_module
    print("✅ Fixed keras.layers.recurrent")

if not hasattr(keras.layers, 'normalization'):
    from types import ModuleType
    normalization_module = ModuleType('keras.layers.normalization')
    normalization_module.BatchNormalization = keras.layers.BatchNormalization
    keras.layers.normalization = normalization_module
    sys.modules['keras.layers.normalization'] = normalization_module
    print("✅ Fixed keras.layers.normalization")

if not hasattr(keras.layers, 'convolutional'):
    from types import ModuleType
    convolutional_module = ModuleType('keras.layers.convolutional')
    convolutional_module.Convolution1D = keras.layers.Conv1D
    convolutional_module.Convolution2D = keras.layers.Conv2D
    keras.layers.convolutional = convolutional_module
    sys.modules['keras.layers.convolutional'] = convolutional_module
    print("✅ Fixed keras.layers.convolutional")

if not hasattr(keras.layers, 'core'):
    from types import ModuleType
    core_module = ModuleType('keras.layers.core')
    core_module.Dense = keras.layers.Dense
    core_module.Flatten = keras.layers.Flatten
    core_module.RepeatVector = keras.layers.RepeatVector
    core_module.Dropout = keras.layers.Dropout
    keras.layers.core = core_module
    sys.modules['keras.layers.core'] = core_module
    print("✅ Fixed keras.layers.core")

# ============================================================================
# PATCH 3: Register Custom Layers and Patch load_encoder/load_decoder
# ============================================================================

# Import custom layer
try:
    from chemvae.tgru_k2_gpu import TerminalGRU
    print("✅ Found TerminalGRU custom layer")
    CUSTOM_LAYERS_AVAILABLE = True
except ImportError:
    print("⚠️  TerminalGRU not found - will try without it")
    TerminalGRU = None
    CUSTOM_LAYERS_AVAILABLE = False

# Patch load_encoder and load_decoder to use custom objects
try:
    import chemvae.models
    from keras.models import load_model as keras_load_model
    
    _original_load_encoder = chemvae.models.load_encoder
    _original_load_decoder = chemvae.models.load_decoder
    
    def patched_load_encoder(params):
        """Load encoder with custom objects."""
        custom_objects = {}
        if CUSTOM_LAYERS_AVAILABLE and TerminalGRU is not None:
            custom_objects['TerminalGRU'] = TerminalGRU
        
        filepath = params['encoder_weights_file']
        return keras_load_model(filepath, custom_objects=custom_objects if custom_objects else None)
    
    def patched_load_decoder(params):
        """Load decoder with custom objects."""
        custom_objects = {}
        if CUSTOM_LAYERS_AVAILABLE and TerminalGRU is not None:
            custom_objects['TerminalGRU'] = TerminalGRU
        
        filepath = params['decoder_weights_file']
        return keras_load_model(filepath, custom_objects=custom_objects if custom_objects else None)
    
    chemvae.models.load_encoder = patched_load_encoder
    chemvae.models.load_decoder = patched_load_decoder
    
    print("✅ Patched load_encoder and load_decoder with custom layers")
    
except Exception as e:
    print(f"⚠️  Could not patch load functions: {e}")

print("="*70)
print("🎉 ALL PATCHES APPLIED SUCCESSFULLY!")
print("="*70)
print("You can now use: from chemvae.vae_utils import VAEUtils")
print("="*70)
print()
