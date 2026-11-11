from classification.metrics import prediction_row


def test_prediction_row_contains_only_filename_and_labels():
    row = prediction_row("/dataset/group/image_01.png"， 0, 2)

    assert row == {
        "filename": "image_01.png",
        "true_label": "Benign"，
        "predicted_label": "Normal"，
    }
