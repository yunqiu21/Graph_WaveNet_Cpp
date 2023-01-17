#include <iostream>
#include <vector>
#include <cmath>
using std::vector;

#define max(x, y) x < y ? y : x
#define min(x, y) x > y ? y : x

#define kNumNodes 20
#define kDropout 0.3
#define kSupportDim0 0
#define kSupportDim1 0
#define kSupportDim2 0
#define kInDim 2
#define kInDim0 4
#define kInDim1 4
#define kInDim2 4
#define kInDim3 4
#define kOutDim 12
#define kResidualChannels 32
#define kDilationChannels 32
#define kSkipChannels 256
#define kEndChannels 512
#define kKernelSize 2
#define kBlocks 4
#define kLayers 2
#define kReceptiveField 13

/********************** helper classes **********************/

class BatchNorm2d {
  
};

class Conv1d {
  public:
  Conv1d(int in_channels, int out_channels, int kernelH, int kernelW, int dilation) {
    m_in_channels = in_channels;
    m_out_channels = out_channels;
    m_kernelH = kernelH;
    m_kernelW = kernelW;
    m_dilation = dilation;
    m_weight = new float[m_kernelH][m_kernelW];
    m_bias = new float[m_out_channels];
  }

  ~Conv1d() {
    delete[] m_weight;
    delete[] m_bias;
  }

  void forward(float* input,  int inW,  int minibatch,
               float* output, int outW) {
    /* convolution */
    // input: [minibatch][m_in_channels][inW]
    // output: [minibatch][m_out_channels][inW-m_dilation(m_kernelW-1)]
    for (int m = 0; m < minibatch; m++) {
      /* set bias */
      for (int i = 0; i < m_out_channels; i++) {
        for (int w = 0; w < outW; w++) {
          output[m][i][w] = m_bias[i];
        }
      }
      /* convolution */
      for (int i = 0; i < m_out_channels; i++) {
        for (int j = 0; j < m_in_channels; j++) {
          for (int w = 0; w < outW; w++) {
            for (int q = 0; q < m_kernelW; q++) {
              output[m][i][w] += input[m][j][w+q*d] * m_weight[i][j][q];
            }
          }
        }
      }
    }
  }

  private:
  int m_in_channels;
  int m_out_channels;
  int m_kernelH;
  int m_kernelW;
  int m_dilation;
  float* m_weight;
  float* m_bias;
};

class Conv2d {  
  public:
  Conv2d(int in_channels, int out_channels, int kernelH, int kernelW, int dilation) {
    m_in_channels = in_channels;
    m_out_channels = out_channels;
    m_kernelH = kernelH;
    m_kernelW = kernelW;
    m_dilation = dilation;
    m_weight = new float[m_kernelH][m_kernelW];
    m_bias = new float[m_out_channels];
  }

  ~Conv2d() {
    delete[] m_weight;
    delete[] m_bias;
  }

  void forward(float* input,  int inH,  int inW,  int minibatch,
               float* output, int outH, int outW) {
    /* convolution */
    // input: [minibatch][m_in_channels][inH][inW]
    // output: [minibatch][m_out_channels][inH-m_dilation(m_kernelH-1)][inW-m_dilation(m_kernelW-1)]
    for (int m = 0; m < minibatch; m++) {
      /* set bias */
      for (int i = 0; i < m_out_channels; i++) {
        for (int h = 0; h < outH; h++) {
          for (int w = 0; w < outW; w++) {
            output[m][i][h][w] = m_bias[i];
          }
        }
      }
      /* convolution */
      for (int i = 0; i < m_out_channels; i++) {
        for (int j = 0; j < m_in_channels; j++) {
          for (int h = 0; h < outH; h++) {
            for (int w = 0; w < outW; w++) {
              for (int p = 0; p < m_kernelH; p++) {
                for (int q = 0; q < m_kernelW; q++) {
                  output[m][i][h][w] += input[m][j][h+p*d][w+q*d] * m_weight[i][j][p][q];
                }
              }
            }
          }
        }
      }
    }
  }

  private:
  int m_in_channels;
  int m_out_channels;
  int m_kernelH;
  int m_kernelW;
  int m_dilation;
  float m_weight[m_kernelH][m_kernelW];
  float m_bias[m_out_channels];
};

/******************* helper functions ******************/

void leaky_relu(float* input, int size) {
  for (int i = 0; i < size; i++) {
    input[i] = max(input[i], 0) + 0.01 * min(input[i], 0);
  }
}

void relu(float* input, int size) {
  for (int i = 0; i < size; i++) {
    input[i] = max(input[i], 0);
  }
}

void softmax_dim1(float* input, int dim0, int dim1) {
  for (int i = 0; i < dim0; i++) {
    float exp_sum = 0;
    for (int j = 0; j < dim1; j++) {
      exp_sum += exp(input[i][j]);
    }
    for (int j = 0; j < dim1; j++) {
      input[i][j] = exp(input[i][j]) / exp_sum;
    }
  }
}

void sigmoid(float* input, int size) {
  for (int i = 0; i < size; i++) {
    input[i] = 1 / (1 + exp(-input[i]));
  }
}

void tanh(float* input, int size) {
  for (int i = 0; i < size; i++) {
    input[i] = tanh(input[i]);
  }
}

void matmul(float* mat1, int mat1H, int mat1W, 
            float* mat2, int mat2H, int mat2W,
            float* output) {
  if (mat1W != mat2H) {
    // throw error
    return;
  }
  for (int i = 0; i < mat1H; i++) {
    for (int k = 0; k < mat2W; k++) {
      output[i][k] = 0;
    }
  }
  // matrix multiplication
  for (int i = 0; i < mat1H; i++) {
    for (int k = 0; k < mat2W; k++) {
      for (int j = 0; j < mat1W; j++) {
        output[i][k] += mat1[i][j] * mat2[j][k];
      }
    }
  }
}

// element-wise matrix addition
void elem_add(float* arr1, float* arr2, float* output, 
              int dim0, int dim1, int dim2, int dim3) {
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
      for (int k = 0; k < dim2; k++) {
        for (int l = 0; l < dim3; l++) {
          output[i][j][k][l] = arr1[i][j][k][l] + arr2[i][j][k][l];
        }
      }
    }
  }
}

// element-wise matrix multiplication
void elem_add(float* arr1, float* arr2, float* output, 
              int dim0, int dim1, int dim2, int dim3) {
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
      for (int k = 0; k < dim2; k++) {
        for (int l = 0; l < dim3; l++) {
          output[i][j][k][l] = arr1[i][j][k][l] * arr2[i][j][k][l];
        }
      }
    }
  }
}

class GraphWaveNet {
 public:
  GraphWaveNet(float dropout=0.3, float* supports=NULL, bool do_graph_conv=true,
         bool addaptadj=true, float* aptinit=NULL, int in_dim=2, int out_dim=12,
         bool cat_feat_gc=false, int apt_size=10);
  ~GraphWaveNet();
  float* forward(float* input);
 private:
  int m_num_nodes;
  float m_dropout;
  float* m_supports; // an array of 2d arrays
  bool m_do_graph_conv;
  bool m_addaptadj;
  float* m_aptinit;
  int m_in_dim;
  int m_out_dim;
  int m_cat_feat_gc;
  int m_skip_channels;
  int m_end_channels;
  int m_apt_size;

  float* m_x;
  int m_receptive_field;
  int m_num_timesteps;

  float* m_dil_convs;

  float m_nodevec1[kNumNodes][10];
  float m_nodevec2[10][kNumNodes];
};

GraphWaveNet::GraphWaveNet(int num_nodes, float dropout, float* supports, bool do_graph_conv,
         bool addaptadj, float* aptinit, int in_dim, int out_dim,
         int residual_channels, int dilation_channels, bool cat_feat_gc,
         int skip_channels, int end_channels, int kernel_size, int blocks, int layers,
         int apt_size) {
  // set member fields
  m_num_nodes = num_nodes;
  m_dropout = dropout;
  m_supports = supports;
  m_do_graph_conv = do_graph_conv;
  m_addaptadj = addaptadj;
  m_aptinit = aptinit;
  m_cat_feat_gc = cat_feat_gc;
  m_apt_size = apt_size;

  int receptive_field = 1;
  for (int b = 0; b < kBlocks; b++) {
    int additional_scope = kKernelSize - 1;
    int D = 1;
    for (int i = 0; i < kLayers; i++) {
      // dilated convolutions
      // self.filter_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
      // self.gate_convs.append(Conv1d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
      m_dil_convs[b * kLayers + i] = D;
      D *= 2;
      receptive_field += additional_scope;
      additional_scope *= 2;
    }
  }
  m_receptive_field = receptive_field;


  /* nodevec */
  std::default_random_engine generator;
  std::normal_distribution<float> distribution(1.0, 0.0);

  if (gcn_bool && addaptadj) {
    if (!aptinit) {
      if (!supports) {
        m_supports = nullptr;
      }
      for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < 10; j++) {
          float number = distribution(generator);
          m_nodevec1[i][j] = number;
        }
      }
      for (int i = 0; i < 10; i++) {
        for (int j = 0; j < num_nodes; j++) {
          float number = distribution(generator);
          m_nodevec2[i][j] = number;
        }
      }
    } else {
      if (!supports) {
        m_supports = nullptr;
      }
      // todo
      m, p, n = torch.svd(aptinit)
      m_nodevec1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
      m_nodevec2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
    }
  }
}

float* GraphWaveNet::forward(float* input) {
  int xDim3 = max(kInDim3, kReceptiveField);
  float x[kInDim0][kInDim1][kInDim2][xDim3];
  if (kInDim3 < kReceptiveField) {
    /* pad */
    for (int i = 0; i < kInDim0; i++) {
      for (int j = 0; j < kInDim1; j++) {
        for (int k = 0; k < kInDim2; k++) {
          for (int l = 0; l < kReceptiveField - kInDim3; l++) {
            x[i][j][k][l] = 0;
          }
          for (int l = kReceptiveField - kInDim3; l < kReceptiveField; l++) {
            x[i][j][k][l] = input[i][j][k][l-(kReceptiveField-kInDim3)];
          }
        }
      }
    }
  } else {
    for (int i = 0; i < kInDim0; i++) {
      for (int j = 0; j < kInDim1; j++) {
        for (int k = 0; k < kInDim2; k++) {
          for (int l = 0; l < kInDim3; l++) {
            x[i][j][k][l] = input[i][j][k][l];
          }
        }
      }
    }
  }

  int skip = 0;
  
  // calculate the current adaptive adj matrix once per iteration
  float new_supports[kSupportDim0][kSupportDim1][kSupportDim2];
  if (m_addaptadj) {
    adp = softmax_dim1(relu(matmul(m_nodevec1, m_nodevec2)), kNumNodes, kNumNodes);
    // new_supports = self.supports + [adp]
    for (int i = 0; i < kSupportDim0 + 1; i++) {
      for (int j = 0; j < kSupportDim1; j++) {
        for (int k = 0; k < kSupportDim2; k++) {
          if (i < kSupportDim0) {
            new_supports[i][j][k] = m_supports[i][j][k];
          } else {
            new_supports[i][j][k] = adp[j][k];
          }
        }
      }
    }
  }


  for (int i = 0; i < kBlocks * kLayers; i++) {
    float* residual = x; // todo: make a shallow copy
    int D = 1 << (i % kLayers);
    int filterH = kInDim2 - 1 * (1 - 1);
    int filterW = xDim3 - D * (kKernelSize - 1);
    float filter[kInDim0][kDilationChannels][filterH][filterW];
    tanh(Conv2d(kResidualChannels, kDilationChannels, 1, kKernelSize, D, residual, filter));
    float gate[kInDim0][kDilationChannels][filterH][filterW];
    float* gate = sigmoid(Conv1d(kResidualChannels, kDilationChannels, 1, kKernelSize, D, residual, gate));
    float x_tmp[kInDim0][kDilationChannels][filterH][filterW];
    elem_mul(filter, gate, );
    x = x_tmp;

    /* skip connection */ 
    // dimension of s: [in0][m_skip_channels][iW-D(kW-1)]
    float* s = Conv1d(kDilationChannels, m_skip_channels, 1, 1, 1, x);
    if (i > 0) {
      skip = skip[:,:,:,-s.size(3):]; // todo
    }
    skip = elem_add(s, skip);

    /* residual */
    if (m_gcn_bool && m_supports) {
      if (m_addaptadj) {

      } else {

      }
    } else {
      Conv1d(kDilationChannels, kResidualChannels, 1, 1, 1, x, x_tmp);
    }
    x = x_tmp;

    x = x + residual[:,:,:,-x.size(3):]; // todo
    x = BatchNorm2d(kResidualChannels, x);
  }

  x1 = relu(skip);
  float x2[kInDim0][kEndChannels][][];
  relu(Conv2d(kSkipChannels, kEndChannels, 1, 1, x1, x2));
  float x3[kInDim0][kOutDim][][];
  relu(Conv2d(kEndChannels, kOutDim, 1, 1, x2, x3));
  return x;
}