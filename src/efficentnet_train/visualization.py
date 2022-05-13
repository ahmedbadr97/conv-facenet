import matplotlib.pyplot as plt


def triplet_prediction_visualization(model, anchor, positive, negative,comparison_method, transform=None, expand_dim=True):
    fig, ax = plt.subplots(1, 3)

    if transform is not None:
        anchor_tensor, positive_tensor, negative_tensor = transform(anchor), transform(positive), transform(negative)
    else :
        anchor_tensor, positive_tensor, negative_tensor =anchor, positive, negative
    if expand_dim:
        anchor_tensor, positive_tensor, negative_tensor = anchor_tensor.unsqueeze(1), positive_tensor.unsqueeze(1), negative_tensor.unsqueeze(1)

    anchor_vector=model(anchor_tensor)[0]
    positive_vector=model(positive_tensor)[0]
    negative_vector=model(negative_tensor)[0]

    if transform is None:
        #images already tensors
        if not expand_dim:
            anchor, positive, negative=anchor.squeeze(), positive.squeeze(), negative.squeeze()
        anchor, positive, negative=anchor.numpy().transpose([1,2,0]),positive.numpy().transpose([1,2,0]),negative.numpy().transpose([1,2,0])
    ax[0].set_title("anchor")
    ax[0].imshow(anchor)

    ax[1].set_title("positive")
    ax[1].imshow(positive)

    ax[2].set_title("negative")
    ax[2].imshow(negative)
    fig.show()
    print(f"anchor vs Positive = {comparison_method(anchor_vector.numpy(),positive_vector.numpy())}")
    print(f"anchor vs negative = {comparison_method(anchor_vector.numpy(),negative_vector.numpy())}")


