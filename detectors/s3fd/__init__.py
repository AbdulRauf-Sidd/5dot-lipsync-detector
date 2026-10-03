import time
import numpy as np
import cv2
import torch
from torchvision import transforms
from .nets import S3FDNet
from .box_utils import nms_

PATH_WEIGHT = './detectors/s3fd/weights/sfd_face.pth'
img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')


class S3FD():

    def __init__(self, device='cuda', weight_path=None):

        tstamp = time.time()
        self.device = device

        print('[S3FD] loading with', self.device)
        self.net = S3FDNet(device=self.device).to(self.device)
        state_dict = torch.load(weight_path or PATH_WEIGHT, map_location=self.device, weights_only=True)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        print('[S3FD] finished loading (%.4f sec)' % (time.time() - tstamp))
    
    @staticmethod
    def preprocess(image, scale):
        """RGB HxWx3 uint8 -> normalized 3xH'xW' float32, ready to batch."""
        scaled_img = cv2.resize(image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        scaled_img = np.swapaxes(scaled_img, 1, 2)
        scaled_img = np.swapaxes(scaled_img, 1, 0)
        scaled_img = scaled_img[[2, 1, 0], :, :]
        scaled_img = scaled_img.astype('float32')
        scaled_img -= img_mean
        scaled_img = scaled_img[[2, 1, 0], :, :]
        return scaled_img

    def detect_batch(self, batch, orig_w, orig_h, conf_th=0.8):
        """
        Run one forward pass over a stack of same-sized preprocessed images
        (see preprocess). Returns a list of per-image (N, 5) arrays of
        [x1, y1, x2, y2, score] in original-image pixels, before cross-scale NMS.
        """
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
            detections = self.net(x, conf_thresh=conf_th).cpu().numpy()

        scale = np.array([orig_w, orig_h, orig_w, orig_h], dtype=np.float32)
        results = []
        for b in range(detections.shape[0]):
            per_image = []
            for i in range(detections.shape[1]):
                d = detections[b, i]
                d = d[d[:, 0] > conf_th]
                if len(d):
                    per_image.append(np.hstack([d[:, 1:] * scale, d[:, :1]]))
            results.append(np.vstack(per_image) if per_image else np.empty(shape=(0, 5)))
        return results

    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        w, h = image.shape[1], image.shape[0]

        bboxes = np.vstack(
            [np.empty(shape=(0, 5))]
            + [self.detect_batch(self.preprocess(image, s)[np.newaxis], w, h, conf_th)[0] for s in scales]
        )

        keep = nms_(bboxes, 0.1)
        return bboxes[keep]
