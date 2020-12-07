class Results:

    def __init__(self,
                 anomaly_detector_name,
                 losses=None,
                 reconstructed_imgs=None):
        self.anomaly_detector_name = anomaly_detector_name,
        self.losses = losses,
        self.reconstructed_imgs = reconstructed_imgs

    def get_losses(self):
        return self.losses

    def get_reconstructed_imgs(self, number=None, direction='top'):
        if number is None:
            return self.reconstructed_imgs
        else:
            if direction == 'top':
                return self.reconstructed_imgs[:number]
            elif direction == 'bottom':
                return self.reconstructed_imgs[number:]
