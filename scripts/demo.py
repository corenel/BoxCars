# -*- coding: utf-8 -*-
# other imports
import os
import sys
import time

import numpy as np

import _init_paths
import cv2
from boxcars_dataset import BoxCarsDataset
from boxcars_image_transformations import unpack_3DBB
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import AveragePooling2D, Dense, Dropout, Flatten
from keras.models import Model, load_model
from keras.optimizers import SGD
from scipy.misc import imread, imresize, imsave
# this should be soon to prevent tensorflow initialization with -h parameter
from utils import parse_args


def init(args):
    """Init dataset and model."""
    # initialize dataset
    if args.estimated_3DBB is None:
        dataset = BoxCarsDataset(load_split="hard", load_atlas=True)
    else:
        dataset = BoxCarsDataset(load_split="hard", load_atlas=True,
                                 use_estimated_3DBB=True,
                                 estimated_3DBB_path=args.estimated_3DBB)

    # get optional path to load model
    model = None
    for path in [args.eval, args.resume]:
        if path is not None:
            print("Loading model from %s" % path)
            model = load_model(path)
            break

    # construct the model as it was not passed as an argument
    if model is None:
        print("Initializing new %s model ..." % args.train_net)
        if args.train_net in ("ResNet50", ):
            base_model = ResNet50(weights='imagenet',
                                  include_top=False,
                                  input_shape=(224, 224, 3))
            x = Flatten()(base_model.output)

        if args.train_net in ("VGG16", "VGG19"):
            if args.train_net == "VGG16":
                base_model = VGG16(weights='imagenet',
                                   include_top=False,
                                   input_shape=(224, 224, 3))
            elif args.train_net == "VGG19":
                base_model = VGG19(weights='imagenet',
                                   include_top=False,
                                   input_shape=(224, 224, 3))
            x = Flatten()(base_model.output)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dropout(0.5)(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dropout(0.5)(x)

        if args.train_net in ("InceptionV3", ):
            base_model = InceptionV3(weights='imagenet',
                                     include_top=False,
                                     input_shape=(224, 224, 3))
            output_dim = int(base_model.outputs[0].get_shape()[1])
            x = AveragePooling2D((output_dim, output_dim), strides=(
                output_dim, output_dim), name='avg_pool')(base_model.output)
            x = Flatten()(x)

        predictions = Dense(dataset.get_number_of_classes(),
                            activation='softmax')(x)
        model = Model(input=base_model.input,
                      output=predictions,
                      name="%s%s" % (
                          args.train_net,
                          {True: "_estimated3DBB",
                           False: ""}[args.estimated_3DBB is not None]))
        optimizer = SGD(lr=args.lr, decay=1e-4, nesterov=True)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=["accuracy"])

    print("Model name: %s" % (model.name))
    if args.estimated_3DBB is not None and "estimated3DBB" not in model.name:
        print("ERROR: using estimated 3DBBs "
              "with model trained on original 3DBBs")
        sys.exit(1)
    if args.estimated_3DBB is None and "estimated3DBB" in model.name:
        print("ERROR: using model trained on estimated 3DBBs "
              "and running on original 3DBBs")
        sys.exit(1)

    args.output_final_model_path = os.path.join(
        args.cache, model.name, "final_model.h5")
    args.snapshots_dir = os.path.join(args.cache, model.name, "snapshots")
    args.tensorboard_dir = os.path.join(args.cache, model.name, "tensorboard")

    dataset.initialize_data("test")

    return model, dataset


def eval(args, dataset, model, image):
    """Evaluate the model."""

    start_time = time.time()

    # image pre-processing
    image = imresize(image, (224, 224))
    image = (image.astype(np.float32) - 116) / 128.
    image = np.array([image])

    pred = model.predict(image)

    # clena the terminal
    print("\033c")

    # print fps
    elapsed_time = time.time() - start_time
    fps = 1000. / elapsed_time
    print("fps: {}".format(fps))

    # print predictions
    pred = np.squeeze(pred)
    print(pred.shape)
    pred_top5 = pred.argsort()[-5:][::-1]
    print(pred_top5.shape)
    for pred_idx in pred_top5:
        cls_name = dataset.get_name_of_classes(pred_idx)
        cls_prob = pred[pred_idx]
        print("[{}]:name=[{}], prob={}".format(pred_idx, cls_name, cls_prob))


if __name__ == "__main__":
    args = parse_args(["ResNet50", "VGG16", "VGG19", "InceptionV3"])
    model, dataset = init(args)

    if args.demo is not None and os.path.exists(args.demo):
        print(args.demo)
        cap = cv2.VideoCapture(args.demo)

    # Check if camera opened successfully
    if (cap.isOpened() is False):
        print("Error opening video stream or file")

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is True:
            # Display the resulting frame
            # cv2.imshow('Frame', frame)
            eval(args, dataset, model, frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    # cv2.destroyAllWindows()
