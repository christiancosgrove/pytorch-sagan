import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F

# TODO: check contiguous in THNN
# TODO: use separate backend functions?
class _ConditionalBatchNorm(Module):

    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ConditionalBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_classes, num_features))
            self.bias = Parameter(torch.Tensor(num_classes, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_classes, num_features))
            self.register_buffer('running_var', torch.ones(num_classes, num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input, label):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean[label], self.running_var[label], self.weight[label], self.bias[label],
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self,
                              state_dict, prefix, strict,
                              missing_keys, unexpected_keys, error_msgs):
        try:
            version = state_dict._metadata[prefix[:-1]]['version']
        except (AttributeError, KeyError):
            version = None

        if version is None and self.track_running_stats:
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                # Add the missing num_batches_tracked counter
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_ConditionalBatchNorm, self)._load_from_state_dict(
            state_dict, prefix, strict,
            missing_keys, unexpected_keys, error_msgs)


class ConditionalBatchNorm2d(_ConditionalBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
