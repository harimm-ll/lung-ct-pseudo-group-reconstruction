from classification.backbone import FineTunedResNet50


def test_only_last_bottleneck_non_bn_parameters_are_trainable():
    model = FineTunedResNet50(pretrained=False)
    names = model.trainable_parameter_names()
    assert names
    assert all(name.startswith("model.layer4.2.") for name in names)
    assert all("bn" not in name for name in names)

