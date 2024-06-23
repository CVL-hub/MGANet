r""" Visualize model predictions """
import os

from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from . import util


class Visualizer:

    @classmethod
    def initialize(cls, visualize):
        cls.visualize = visualize
        if not visualize:
            return

        cls.colors = {'red': (255, 50, 50), 'blue': (102, 140, 255)}
        for key, value in cls.colors.items():
            cls.colors[key] = tuple([c / 255 for c in cls.colors[key]])

        cls.mean_img = [0.485, 0.456, 0.406]
        cls.std_img = [0.229, 0.224, 0.225]
        cls.to_pil = transforms.ToPILImage()
        cls.vis_path = './vis/LoveDA/MGANet_split0/'
        if not os.path.exists(cls.vis_path): os.makedirs(cls.vis_path)

    @classmethod
    def visualize_prediction_batch(cls, spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, out_aux_b, qry_tmp_mask_b, supp_tmp_mask_b,
                                                  prior_mask_b, mask_co_b, mask_aux_b, cls_id_b, batch_idx, iou_b=None):
        spt_img_b = util.to_cpu(spt_img_b)
        spt_mask_b = util.to_cpu(spt_mask_b)
        qry_img_b = util.to_cpu(qry_img_b)
        qry_mask_b = util.to_cpu(qry_mask_b)
        pred_mask_b = util.to_cpu(pred_mask_b)
        qry_tmp_mask_b = util.to_cpu(qry_tmp_mask_b)
        supp_tmp_mask_b = util.to_cpu(supp_tmp_mask_b)
        prior_mask_b = util.to_cpu(prior_mask_b)
        mask_co_b = util.to_cpu(mask_co_b)
        mask_aux_b = util.to_cpu(mask_aux_b)
        out_aux_b = util.to_cpu(out_aux_b)
        cls_id_b = util.to_cpu(cls_id_b)

        for sample_idx, (spt_img, spt_mask, qry_img, qry_mask, pred_mask, out_aux, qry_tmp_mask, supp_tmp_mask, prior_mask, mask_co, mask_aux, cls_id) in \
                enumerate(zip(spt_img_b, spt_mask_b, qry_img_b, qry_mask_b, pred_mask_b, out_aux_b, qry_tmp_mask_b, supp_tmp_mask_b,
                                                  prior_mask_b, mask_co_b, mask_aux_b, cls_id_b)):
            iou = iou_b if iou_b is not None else None
            #cls.visualize_prediction_binary(spt_img, spt_mask, qry_img, qry_mask, pred_mask1, pred_mask2, corr_mask, z, cls_id, batch_idx, sample_idx, True, iou)
            cls.visualize_prediction(spt_img, spt_mask, qry_img, qry_mask, pred_mask, out_aux, qry_tmp_mask, 
                                     supp_tmp_mask, prior_mask, mask_co, mask_aux, cls_id, batch_idx, sample_idx, True, iou)

    @classmethod
    def to_numpy(cls, tensor, type):
        if type == 'img':
            return np.array(cls.to_pil(cls.unnormalize(tensor))).astype(np.uint8)
        elif type == 'mask':
            return np.array(tensor).astype(np.uint8)
        else:
            raise Exception('Undefined tensor type: %s' % type)

    @classmethod
    def visualize_prediction(cls, spt_imgs, spt_masks, qry_img, qry_mask, pred_mask, out_aux, qry_tmp_mask, supp_tmp_mask,
                             prior_mask, mask_co, mask_aux, cls_id, batch_idx, sample_idx, label, iou=None):
        iou = iou.item() if iou else 0.0
        file_path = cls.vis_path + '%d_%d_class-%d_iou-%.2f' % (batch_idx, sample_idx, cls_id, iou)
        os.makedirs(file_path, exist_ok=True)
        spt_color = cls.colors['blue']
        qry_color = cls.colors['red'] #记得修改回去
        pred_color = cls.colors['red'] 
        # support图片和query图片
        spt_imgs = [cls.to_numpy(spt_img, 'img') for spt_img in spt_imgs]
        #spt_imgs = [Image.fromarray(spt_img) for spt_img in spt_imgs]
        spt_masks = [cls.to_numpy(spt_mask, 'mask') for spt_mask in spt_masks]
        spt_masked_pils = [Image.fromarray(cls.apply_mask(spt_img, spt_mask, spt_color)) for spt_img, spt_mask in zip(spt_imgs, spt_masks)]

        qry_img = cls.to_numpy(qry_img, 'img')
        #qry_img = Image.fromarray(qry_img)
        qry_mask = cls.to_numpy(qry_mask, 'mask')
        pred_mask = cls.to_numpy(pred_mask, 'mask')
        out_aux = cls.to_numpy(out_aux, 'mask')
        pred_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), pred_mask.astype(np.uint8), pred_color))
        qry_masked_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), qry_mask.astype(np.uint8), qry_color))
        out_aux_pil = Image.fromarray(cls.apply_mask(qry_img.astype(np.uint8), out_aux.astype(np.uint8), qry_color))

        for idx, spt_masked in enumerate(spt_masked_pils):
            spt_masked.save(file_path + "/support%d" % (idx) + ".jpg")
        qry_masked_pil.save(file_path + "/query" + ".jpg")
        pred_masked_pil.save(file_path + "/main_pred" + ".jpg")
        out_aux_pil.save(file_path + "/aux_pred" + ".jpg")
        # # support mask与 query mask
        # spt_masks = [cls.to_numpy(spt_mask, 'mask') for spt_mask in spt_masks]
        # spt_masks = [Image.fromarray((spt_mask * 255).astype(np.uint8), 'L') for spt_mask in spt_masks]

        # qry_mask = cls.to_numpy(qry_mask, 'mask')
        # qry_mask = Image.fromarray((qry_mask * 255).astype(np.uint8), 'L')

        # # pred mask
        # pred_mask = cls.to_numpy(pred_mask, 'mask')
        # pred_mask = Image.fromarray((pred_mask * 255).astype(np.uint8), 'L')

        # # qry_tmp_mask & supp_tmp_mask
        # qry_tmp_mask = cls.to_numpy(qry_tmp_mask, 'mask')
        # qry_tmp_mask = Image.fromarray((qry_tmp_mask * 255).astype(np.uint8), 'L')

        # supp_tmp_mask = cls.to_numpy(supp_tmp_mask, 'mask')
        # supp_tmp_mask = Image.fromarray((supp_tmp_mask * 255).astype(np.uint8), 'L')

        # # mask_co & mask_aux
        # mask_co = cls.to_numpy(mask_co, 'mask')
        # mask_co = Image.fromarray((mask_co * 255).astype(np.uint8), 'L')

        # mask_aux = cls.to_numpy(mask_aux, 'mask')
        # mask_aux = Image.fromarray((mask_aux * 255).astype(np.uint8), 'L')

        # for idx, supp_img in enumerate(spt_imgs):
        #     supp_img.save(file_path + "/supp%d" % (idx) + ".jpg")
        # for idx, supp_mask in enumerate(spt_masks):
        #     supp_mask.save(file_path + "/supp_mask%d" % (idx) + ".jpg")
        # qry_img.save(file_path + "/qry" + ".jpg")
        # qry_mask.save(file_path + "/qry_mask" + ".jpg")

        # pred_mask.save(file_path + "/pred_mask" + ".jpg")

        # mask_co.save(file_path + "/mask_co" + ".jpg")
        # mask_aux.save(file_path + "/mask_aux" + ".jpg")

        # supp_tmp_mask.save(file_path + "/supp_tmp_mask" + ".jpg")
        # qry_tmp_mask.save(file_path + "/qry_tmp_mask" + ".jpg")

        # plt.imshow(prior_mask, cmap='turbo', interpolation='nearest')
        # plt.axis('off')
        # plt.savefig(file_path + "/prior_mask" + ".jpg", bbox_inches='tight', pad_inches=0)
        # #spt_pils = [cls.to_pil(spt_img) for spt_img in spt_imgs]
        # spt_masks = [cls.to_numpy(spt_mask, 'mask') for spt_mask in spt_masks]
        # spt_ori = [Image.fromarray(spt_img) for spt_img in spt_imgs]
        # qry_img = cls.to_numpy(qry_img, 'img')
        # qry_img = Image.fromarray(qry_img)
        # qry_mask = cls.to_numpy(qry_mask, 'mask')
        # pred_mask = cls.to_numpy(pred_mask, 'mask')
        # for idx, s in enumerate(spt_ori):
        #     s.save(file_path + "/spt%d" % (idx) + ".jpg")
        # qry_img.save(file_path + "/qry" + ".jpg")    

    @classmethod
    def visualize_prediction_binary(cls, spt_imgs, spt_masks, qry_img, qry_mask, pred_mask1, pred_mask2, corr_masks, z, cls_id, batch_idx, sample_idx, label, iou=None):
        iou = iou.item() if iou else 0.0
        file_path = cls.vis_path + '%d_%d_class-%d_iou-%.2f' % (batch_idx, sample_idx, cls_id, iou)
        os.makedirs(file_path, exist_ok=True)
        # spt_imgs = [cls.to_numpy(spt_img, 'img') for spt_img in spt_imgs]
        # spt_masks = [cls.to_numpy(spt_mask, 'mask') for spt_mask in spt_masks]
        # spt_masks = [Image.fromarray((spt_mask * 255).astype(np.uint8), 'L') for spt_mask in spt_masks]
        # spt_imgs = [Image.fromarray(spt_img) for spt_img in spt_imgs]
        # qry_img = cls.to_numpy(qry_img, 'img')
        # qry_mask = cls.to_numpy(qry_mask, 'mask')
        # qry_mask = Image.fromarray((qry_mask * 255).astype(np.uint8), 'L')
        # pred_mask1 = cls.to_numpy(pred_mask1, 'mask')
        # pred_mask2 = cls.to_numpy(pred_mask2, 'mask')
        # pred_mask1 = Image.fromarray((pred_mask1 * 255).astype(np.uint8), 'L')
        # pred_mask2 = Image.fromarray((pred_mask2 * 255).astype(np.uint8), 'L')
        # qry_img = Image.fromarray(qry_img)
        #merged_pil = cls.merge_image_pair(spt_imgs + [qry_img])
        #merged_pil.save(cls.vis_path + '%d_%d_class-%d_iou-%.2f' % (batch_idx, sample_idx, cls_id, iou) + '.jpg')
        # for idx, spt_img in enumerate(spt_imgs):
        #     spt_img.save(file_path + '/spt%d' % (idx) + ".jpg")
        # for idx, spt_mask in enumerate(spt_masks):
        #     spt_mask.save(file_path + '/spt_mask%d' % (idx) + ".jpg")
        # for idx, corr_mask in enumerate(corr_masks):
        #      plt.imshow(corr_mask, cmap='turbo', interpolation='nearest')
        #      plt.axis('off')
        #      plt.savefig(file_path + "/corr_mask%d" %(idx) + ".png", bbox_inches='tight', pad_inches=0)
        for idx, z_ in enumerate(z):
            plt.imshow(z_, cmap='turbo', interpolation='nearest')
            plt.axis('off')
            plt.savefig(file_path + "/similarity_map%d" %(idx) + ".png", bbox_inches='tight', pad_inches=0)
        # qry_img.save(file_path + '/qry' + ".jpg") 
        # qry_mask.save(file_path + "/qry_mask" + ".jpg")
        # pred_mask1.save(file_path + "/aux_mask" + ".jpg")
        # pred_mask2.save(file_path + "/main_mask" + ".jpg")   
    @classmethod
    def merge_image_pair(cls, pil_imgs):
        r""" Horizontally aligns a pair of pytorch tensor images (3, H, W) and returns PIL object """

        canvas_width = sum([pil.size[0] for pil in pil_imgs])
        canvas_height = max([pil.size[1] for pil in pil_imgs])
        canvas = Image.new('RGB', (canvas_width, canvas_height))

        xpos = 0
        for pil in pil_imgs:
            canvas.paste(pil, (xpos, 0))
            xpos += pil.size[0]

        return canvas
    
    
    @classmethod
    def apply_mask(cls, image, mask, color, alpha=0.5):
        r""" Apply mask to the given image. """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    @classmethod
    def unnormalize(cls, img):
        img = img.clone()
        for im_channel, mean, std in zip(img, cls.mean_img, cls.std_img):
            im_channel.mul_(std).add_(mean)
        return img
