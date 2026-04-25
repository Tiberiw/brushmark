    # train_transforms_original = transforms.Compose([
    #     transforms.RandomResizedCrop(384, scale=(0.2, 1.0)), #448
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(20),
    #     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     ),
    #     transforms.RandomErasing(p=0.3)
    # ])

    # train_transforms_base = transforms.Compose([
    #     transforms.Resize((384, 384)), #448
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     ),
    # ])

    # train_transforms_random_resized_crop = transforms.Compose([
    #     transforms.RandomResizedCrop(384, scale=(0.2, 1.0)), #448
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     ),
    # ])

    # train_transforms_random_horizontal_flip = transforms.Compose([
    #     transforms.Resize((384, 384)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     ),
    # ])

    # train_transforms_random_rotation = transforms.Compose([
    #     transforms.Resize((384, 384)), #448
    #     transforms.RandomRotation(20),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     ),
    # ])

    # train_transforms_color_jitter = transforms.Compose([
    #     transforms.Resize((384, 384)), #448
    #     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #     transforms.ToTensor(),  
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     ),
    # ])

    # train_transforms_random_erasing = transforms.Compose([
    #     transforms.Resize((384, 384)), #448
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     ),
    #     transforms.RandomErasing(p=0.3)
    # ])

    # train_transforms_no_random_resized_crop = transforms.Compose([
    #     transforms.Resize((384, 384)), #448
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(20),
    #     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     ),
    #     transforms.RandomErasing(p=0.3)
    # ])

    # train_transforms_no_random_horizontal_flip = transforms.Compose([
    #     transforms.RandomResizedCrop(384, scale=(0.2, 1.0)), #448
    #     transforms.RandomRotation(20),
    #     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     ),
    #     transforms.RandomErasing(p=0.3)
    # ])

    # train_transforms_no_random_rotation = transforms.Compose([
    #     transforms.RandomResizedCrop(384, scale=(0.2, 1.0)), #448
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     ),
    #     transforms.RandomErasing(p=0.3)
    # ])

    # train_transforms_no_color_jitter = transforms.Compose([
    #     transforms.RandomResizedCrop(384, scale=(0.2, 1.0)), #448
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(20),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     ),
    #     transforms.RandomErasing(p=0.3)
    # ])

    # train_transforms_no_random_erasing = transforms.Compose([
    #     transforms.RandomResizedCrop(384, scale=(0.2, 1.0)), #448
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(20),
    #     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     ),
    # ])

    # experiment_transforms = {
        # "original": train_transforms_original,
        # "base": train_transforms_base,
        # "random_resized_crop": train_transforms_random_resized_crop,
        # "random_horizontal_flip": train_transforms_random_horizontal_flip,
        # "random_rotation": train_transforms_random_rotation,
        # "color_jitter": train_transforms_color_jitter,
        # "random_erasing": train_transforms_random_erasing,
        # "no_random_resized_crop": train_transforms_no_random_resized_crop,
        # "no_random_horizontal_flip": train_transforms_no_random_horizontal_flip,
        # "no_random_rotation": train_transforms_no_random_rotation,
        # "no_color_jitter": train_transforms_no_color_jitter,
        # "no_random_erasing": train_transforms_no_random_erasing
    # }
