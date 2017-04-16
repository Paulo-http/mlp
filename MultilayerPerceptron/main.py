import sys

from helper import Helper
from mlp import MLP

# Main Class
if __name__ == '__main__':
   helper = Helper()
   for idx in range(len(helper.config)):
      helper.prepare_list(idx)
      print "Initialize MLP with", helper.filename
      helper.outputs = helper.classes
      hidden_dims = 2 * max(helper.attrs, helper.classes)
      for validation in helper.validations:
         training = helper.prepare_training(validation)
         mlp = MLP(training, sys.maxint, (hidden_dims, helper.attrs))
         mlp.train()
         result = 0
         for pattern in validation:
            result +=  mlp.execute(pattern)
         helper.print_result(helper.current_part(validation),
                             (hidden_dims, 0, 0),
                             result/len(validation),
                             mlp.absolute_error,
                             mlp.squared_error,
                             mlp.iteration)