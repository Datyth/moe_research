DATASET_REGISTRY = {}


def register_dataset(name: str):
    def decorator(dataset_class):
        if name in DATASET_REGISTRY:
            raise ValueError(
                f"Dataset '{name}' is already registered."
            )

        DATASET_REGISTRY[name] = dataset_class
        return dataset_class

    return decorator


def build_dataset(config, split, transform = None):
    try:
        dataset_class = DATASET_REGISTRY[config.name]
    except KeyError as error:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(
            f"Unknown dataset '{config.name}'. "
            f"Available: {available}"
        ) from error

    return dataset_class(
        config = config,
        split = split,
        transform = transform,
    )

# train_dataset = build_dataset(
#     dataset_config,
#     split = "train",
#     transform = train_transform,
# )

# test_dataset = build_dataset(
#     dataset_config,
#     split = "test",
#     transform = test_transform,
# )