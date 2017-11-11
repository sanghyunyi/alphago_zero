import numpy as np
import os, pickle, glob, random, sys
import h5py as h5
import json
from collections import namedtuple
from tensorflow.contrib.keras import optimizers as O
from tensorflow.contrib.keras import callbacks as C
from tensorflow.contrib.keras import backend as K
from AlphagoZero.models.policy_value import PolicyValue
from AlphagoZero.preprocessing.preprocessing import Preprocess
from AlphagoZero.util import random_transform
from AlphagoZero import go
from AlphagoZero.preprocessing.preprocessing import BOARD_TRANSFORMATIONS


# the memory
Event = namedtuple('Event', ['state', 'pi', 'reward'])

class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.idx = 0
        self.mem = []

    def add_event(self, event):
        if len(self.mem) < self.capacity:
            self.mem.append(event)
        else:
            self.mem[self.idx] = event
            self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)

def batch_generator_with_buffer(dataset_dir, batch_size, memory_size):
    replay_buffer = Memory(memory_size)
    preprocessor = Preprocess()
    dataset_dir = os.path.join(dataset_dir, '*.pkl')
    dataset_dir_list = glob.glob(dataset_dir)
    dataset_dir_list.sort()
    #print(dataset_dir_list)
    # Initialize memory with data from self_play
    for data_path in dataset_dir_list:
        with open(data_path, 'r') as f:
            data = pickle.load(f)
            f.close()
            for i in range(len(data["state"])):
                replay_buffer.add_event(Event(data["state"][i], data["pi"][i], data["reward"][i]))
    game_size = replay_buffer.sample(1)[0].state.size
    output_dim = preprocessor.output_dim
    state_batch_shape = (batch_size, output_dim, game_size, game_size)
    Xbatch = np.zeros(state_batch_shape)
    Y1batch = np.zeros((batch_size, game_size*game_size + 1)) # for policy
    Y2batch = np.zeros((batch_size, 1)) # for value
    data_path_set = set(dataset_dir_list)
    while True:
        new_data_path_set = set(glob.glob(dataset_dir))
        only_new_data_path_set = new_data_path_set - data_path_set
        data_path_set = new_data_path_set
        for data_path in only_new_data_path_set:
            print('new_data')
            with open(data_path, 'r') as f:
                data = pickle.load(f)
                f.close()
                for i in range(len(data["state"])):
                    replay_buffer.add_event(Event(data["state"][i], data["pi"][i], data["reward"][i]))
        batch_list = replay_buffer.sample(batch_size)
        for batch_idx, batch in enumerate(batch_list):
            transform = random_transform()
            #print('------')
            #print(transform)
            state = preprocessor.state_to_tensor(batch.state,transform)
            state = state.reshape(output_dim, game_size, game_size)
            pass_move_P = 0
            action_board = np.zeros((game_size,game_size))
            for (action, P) in batch.pi:
                if action == go.PASS_MOVE:
                    pass_move_P = P
                else:
                    action_board[action] = P
            action_board = BOARD_TRANSFORMATIONS[transform](action_board)
            action = np.array([pass_move_P]+list(action_board.flatten()))
            #print(batch.pi)
            #print(action)
            #print(batch.reward)
            v = batch.reward
            Xbatch[batch_idx] = state
            Y1batch[batch_idx] = action
            Y2batch[batch_idx] = v
        yield (Xbatch, {"activation_41": Y1batch, "activation_44": Y2batch})


class MetadataWriterCallback(C.Callback):
    def __init__(self, path):
        self.file = path
        self.metadata = {
            "epochs": [],
            "best_epoch": 0
        }

    def on_epoch_end(self, epoch, logs={}):
        # in case appending to logs (resuming training), get epoch number ourselves
        epoch = len(self.metadata["epochs"])

        self.metadata["epochs"].append(logs)

        if "val_loss" in logs:
            key = "val_loss"
        else:
            key = "loss"

        best_loss = self.metadata["epochs"][self.metadata["best_epoch"]][key]
        if logs.get(key) < best_loss:
            self.metadata["best_epoch"] = epoch

        with open(self.file, "w") as f:
            json.dump(self.metadata, f, indent=2)


def run_optimization(cmd_line_args=None):
    """Run training. command-line args may be passed in as a list
    """
    import argparse
    parser = argparse.ArgumentParser(description='Perform optimization on a policy value network.')
    parser.add_argument("--board_size", default=5, type=int, help="Board size")
    parser.add_argument("--model", default='network.json', help="Path to a JSON model file (i.e. from PolicyValue.save_model())")  # noqa: E501
    parser.add_argument("--train_data_directory", default="/./data", help="A .h5 file of training data")
    parser.add_argument("--out_directory", default="/../ckpt/optimized", help="directory where metadata and weights will be saved")
    parser.add_argument("--minibatch", "-B", help="Size of training data minibatches. Default: 16", type=int, default=32)  # noqa: E501
    parser.add_argument("--memory_size", help="Size of replay buffer", type=int, default=50000)
    parser.add_argument("--epoch_length", "-l", help="Number of training examples considered 'one epoch'. Default: # training data", type=int, default=1000)  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    # slightly fancier args
    parser.add_argument("--symmetries", help="Comma-separated list of transforms, subset of noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2", default='noop,rot90,rot180,rot270,fliplr,flipud,diag1,diag2')  # noqa: E501
    # TODO - an argument to specify which transformations to use, put it in metadata

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    args.train_data_directory = os.path.dirname(__file__) + args.train_data_directory
    args.out_directory = os.path.dirname(__file__) + args.out_directory

    # ensure output directory is available
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)

    weight_list = glob.glob(args.out_directory+"/weights.*.hdf5")
    weight_list.sort()
    if len(weight_list) != 0:
        resume = weight_list[-1]
        policy_value = PolicyValue.load_model(os.path.join(args.out_directory, args.model))
        model = policy_value.model
        print(resume)
        model.load_weights(resume)
        start_epoch = int(os.path.basename(resume).split('.')[1])
    else:
        policy_value = PolicyValue(["board"], init_network=True, board=args.board_size)
        policy_value.save_model(os.path.join(args.out_directory, args.model), os.path.join(args.out_directory, 'weights.00000.hdf5'))
        model = policy_value.model
        start_epoch = 0

    if args.verbose:
        if resume:
            print("trying to resume from %s with weights %s" %
                  (args.out_directory, resume))
        else:
            if os.path.exists(args.out_directory):
                print("directory %s exists. any previous data will be overwritten" %
                      args.out_directory)
            else:
                print("starting fresh output directory %s" % args.out_directory)


    # create metadata file and the callback object that will write to it
    meta_file = os.path.join(args.out_directory, "optimization_metadata.json")
    meta_writer = MetadataWriterCallback(meta_file)
    # load prior data if it already exists
    if os.path.exists(meta_file) and resume:
        with open(meta_file, "r") as f:
            meta_writer.metadata = json.load(f)
        if args.verbose:
            print("previous metadata loaded: %d epochs. new epochs will be appended." %
                  len(meta_writer.metadata["epochs"]))
    elif args.verbose:
        print("starting with empty metadata")
    # Record all command line args in a list so that all args are recorded even
    # when training is stopped and resumed.
    meta_writer.metadata["cmd_line_args"] = meta_writer.metadata.get("cmd_line_args", [])
    meta_writer.metadata["cmd_line_args"].append(vars(args))

    # create ModelCheckpoint to save weights every epoch
    checkpoint_template = os.path.join(args.out_directory, "weights.{epoch:07d}.hdf5")
    checkpointer = C.ModelCheckpoint(checkpoint_template)


    # create dataset generators
    train_data_directory = os.path.join(args.train_data_directory, '*.pkl')
    while True:
        data_path_set = set(glob.glob(train_data_directory))
        if len(data_path_set) > 0:
            break
    train_data_generator = batch_generator_with_buffer(args.train_data_directory, args.minibatch, args.memory_size)

    def lr_scheduler(epoch):
        if epoch == 400000:
            K.set_value(model.optimizer.lr, .001)
        elif epoch == 600000:
            K.set_value(model.optimizer.lr, .0001)
        return K.get_value(model.optimizer.lr)

    change_lr = C.LearningRateScheduler(lr_scheduler)
    sgd = O.SGD(lr=.01, momentum=0.9)
    model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=sgd)

    if args.verbose:
        print("STARTING TRAINING")

    while True:
        model.fit_generator(
            generator=train_data_generator,
            epochs = sys.maxint,
            steps_per_epoch=args.epoch_length,
            callbacks=[checkpointer, meta_writer, change_lr],
            initial_epoch = start_epoch
            )
        start_epoch += 1


if __name__ == '__main__':
    run_optimization()
