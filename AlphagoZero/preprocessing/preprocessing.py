import numpy as np
import AlphagoZero.go as go

##
# individual feature functions (state --> tensor) begin here
##
BOARD_TRANSFORMATIONS = {
    "noop": lambda feature: feature,
    "rot90": lambda feature: np.rot90(feature, 1), # counter clock wise 90
    "rot180": lambda feature: np.rot90(feature, 2),
    "rot270": lambda feature: np.rot90(feature, 3),
    "fliplr": lambda feature: np.fliplr(feature),
    "flipud": lambda feature: np.flipud(feature),
    "diag1": lambda feature: np.transpose(feature),
    "diag2": lambda feature: np.fliplr(np.rot90(feature, 1))
}


def get_board(state, transform):
    """A feature encoding WHITE BLACK and EMPTY on separate planes, but plane 0
    always refers to the current player and plane 1 to the opponent
    """
    planes = np.zeros((17, state.size, state.size))
    board_history = state.board_history
    for i in range(8):
        planes[2*i,:,:] = board_history[7-i] == state.current_player
        planes[2*i+1,:,:] = board_history[7-i] == -state.current_player
    if state.current_player == go.BLACK:
        planes[16,:,:] = 1
    else:
        planes[16,:,:] = 0
    planes = np.array([BOARD_TRANSFORMATIONS[transform](plane) for plane in planes])
    return planes




# named features and their sizes are defined here
FEATURES = {
    "board": {
        "size": 17,
        "function": get_board
    }
}

DEFAULT_FEATURES = [
    "board"]

class Preprocess(object):
    """a class to convert from AlphaGo GameState objects to tensors of one-hot
    features for NN inputs
    """

    def __init__(self, feature_list=DEFAULT_FEATURES):
        """create a preprocessor object that will concatenate together the
        given list of features
        """

        self.output_dim = 0
        self.feature_list = feature_list
        self.processors = [None] * len(feature_list)
        for i in range(len(feature_list)):
            feat = feature_list[i].lower()
            if feat in FEATURES:
                self.processors[i] = FEATURES[feat]["function"]
                self.output_dim += FEATURES[feat]["size"]
            else:
                raise ValueError("uknown feature: %s" % feat)

    def state_to_tensor(self, state, transform="noop"):
        """Convert a GameState to a Theano-compatible tensor
        """
        feat_tensors = [proc(state, transform) for proc in self.processors]

        # concatenate along feature dimension then add in a singleton 'batch' dimension
        f, s = self.output_dim, state.size
        return np.concatenate(feat_tensors).reshape((1, f, s, s))
