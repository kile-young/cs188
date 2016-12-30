import numpy as np
from util import raiseNotDefined

def gradient(f, w):
    """YOUR CODE HERE"""
    # np.set_printoptions()
    # print 'w: ', w
    delta = 0.01
    # print "shape: ", np.shape(w), w
    alterations = np.zeros((len(w), len(w)))
    for i in range(len(w)):
      temp = w.copy()
      temp[i] += delta
      alterations[i] = temp

    # print 'alts: ', alterations

    # print 'w: ', w
    orig = f(w)
    # print "orig: ", [orig]
    funcs = [f(a) for a in alterations]
    # print 'true: ', funcs[0]==w

    # print 'funcs: ', funcs
    # print 'eq: ', funcs[0]-orig, funcs[0], orig
    grad = [(fn-orig)/delta for fn in funcs]


    return grad


    # delta = 0.100
    # alterations = np.zeros((len(w), len(w)))
    # for i in range(len(w)):
    #   temp = w.copy()
    #   temp[i] += delta
    #   alterations[i] = temp

    # orig = f(w)
    # funcs = [f(a) for a in alterations]
    # grad = [(fn-orig)/delta for fn in funcs]
   

    # return grad

def sanity_check_gradient():
  """Handy function for debugging your gradient method."""
  def g(w):
    w1 = w[0]
    w2 = w[1]
    return w1 ** 3 * w2 + 3 * w1
  
  print("The print statement below should output approximately [111, 27]")
  print(gradient(g, np.array([3, 4], dtype='f')))

  def loss(self, sample, label, w):
      """sample: np.array of shape(1, numFeatures).
      label:  the correct label of the sample 
      w:      the weight vector under which to calculate loss

      Can interpret loss as the probability of sample being in the correct class when
      classified by a SigmoidNeuron. 

      For numerical accuracy reasons, the loss is expressed as 
      math.log(1/sigmoid) instead of -math.log(sigmoid) as we discussed in class.

      Do not modify this function."""
      z = np.dot(w, sample)

      if label == True:
          return math.log(1.0 + math.exp(-2*z))
      else:
          return math.log(1.0 + math.exp(2*z))


sanity_check_gradient()