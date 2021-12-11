def augmentationFactory(augmentation: str) -> None:
    """[summary]

    Args:
        augmentation (str): [description]

    Returns:
        [type]: [description]
    """    

  if augmentation == 'augment':
        transform = [
            transforms.Resize((400,400)),
            transforms.RandomHorizontalFlip(),
        ]

  elif augmentation == 'noaugment':
          transform = [
            transforms.Resize((400,400)),
        ]

  else: 
      NotImplemented(f"augment parameter {augmentation} not implemented")

  normalize = transforms.Normalize(mean=[0.6475, 0.4907, 0.4165],
                                     std=[0.1875, 0.1598, 0.1460])

  return transforms.Compose(transform + [transforms.ToTensor()])                 #, normalize])       
