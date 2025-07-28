
# ------------------ dataset ------------------------
class OrganoidDataset(Dataset):
    """
    Dataset that returns (img, mask, label).
    When `augment=True`, applies:
        • Random zoom-in + crop (prob. 0.5)         <-- NEW
        • Random horizontal flip
    """
    def __init__(self, img_paths, mask_paths, labels, augment=False,
                 zoom_scale=(0.8, 1.0)):           # <-- new arg
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.labels     = labels
        self.augment    = augment
        self.zoom_scale = zoom_scale

        # separate “resize” and “to-tensor” so we can insert crop beforehand
        self.t_resize_img  = T.Resize(TARGET_SIZE)
        self.t_resize_mask = T.Resize(TARGET_SIZE,
                                      interpolation=T.InterpolationMode.NEAREST)
        self.t_to_tensor   = T.ToTensor()


    def __len__(self):
        return len(self.labels)

    # -------- augmentation helpers ---------
    @staticmethod
    def _flip(img, mask):
        if torch.rand(()) > 0.5:
            img  = T.functional.hflip(img)
            mask = T.functional.hflip(mask)
        return img, mask

    @staticmethod
    def _zoom_crop_pair(img, mask, scale=(0.8, 1.0)):
        """
        Apply the *same* RandomResizedCrop to an RGB image and its mask.
        Aspect-ratio locked to 1.0 ⇒ pure zoom-in/out (no stretching).
        """
        i, j, h, w = T.RandomResizedCrop.get_params(
            img, scale=scale, ratio=(1.0, 1.0))

        img  = T.functional.resized_crop(
            img,  i, j, h, w, size=TARGET_SIZE,
            interpolation=T.InterpolationMode.BILINEAR)

        mask = T.functional.resized_crop(
            mask, i, j, h, w, size=TARGET_SIZE,
            interpolation=T.InterpolationMode.NEAREST)
        return img, mask
    # ---------------------------------------

    def __getitem__(self, idx):
        # ---- load PIL images ----
        img  = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # ---- random zoom-crop (p=0.5) ----
        if self.augment and torch.rand(()) > 0.5:
            img, mask = self._zoom_crop_pair(img, mask,
                                             scale=self.zoom_scale)
        else:
            # deterministic resize path
            img  = self.t_resize_img(img)
            mask = self.t_resize_mask(mask)

        # ---- other augmentations ----
        if self.augment:
            img, mask = self._flip(img, mask)

        # ---- to tensor ----
        img  = self.t_to_tensor(img)
        mask = self.t_to_tensor(mask)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, mask, label
# ---------------------------------------------------
