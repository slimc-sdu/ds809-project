import keras

class myCallback(keras.callbacks.Callback): 
    targetAccuracy = 0.9
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('accuracy') > (self.targetAccuracy)):   
            self.model.stop_training = True