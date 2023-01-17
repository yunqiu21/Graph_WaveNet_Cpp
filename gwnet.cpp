#include <iostream>
#include <vector>
#include <cmath>
using std::vector;

#define max(x, y) x < y ? y : x
#define min(x, y) x > y ? y : x

#define kInDim0 4
#define kInDim1 4
#define kInDim2 4
#define kInDim3 4

#define kKernelSize 2

#define kResidualChannels 3
#define kDilationChannels 3
#define kSkipChannels 3
#define kEndChannels 3
#define kOutDim 3

#define kBlocks 4
#define kLayers 2

#define kReceptiveField 10

#define kSupportDim0 0
#define kSupportDim1 0
#define kSupportDim2 0

#define kNumNodes 20

void BatchNorm2d() {
  
}

void Conv1d(int in_channel, int out_channel, float* bias, float* weight,
        int kernelW, int d, float* data, int inW) {
  // convolution
  // input dimension: [minibatch][in_channels][iW]
  // output dimension: [minibatch][out_channels][iW-D(kW-1)]

  /* calculate output dimension */
  int outW = inW - d * (kernelW - 1);
  float* output;
  for (int m = 0; m < minibatch; m++) {
    /* set bias */
    for (int i = 0; i < out_channel; i++) {
      for (int w = 0; w < outW; w++) {
        output[m][i][w] = bias[i];
      }
    }
    /* convolution */
    for (int i = 0; i < out_channel; i++) {
      for (int j = 0; j < in_channel; j++) {
        for (int w = 0; w < outW; w++) {
          for (int q = 0; q < kernelW; q++) {
            output[m][i][w] += data[m][j][w+q*d] * weight[i][j][q];
          }
        }
      }
    }
  }
}

void Conv2d(int in_channel, int out_channel, float* bias, float* weight,
        int kernelH, int kernelW, int d, float* data, int inH, int inW, float* output) {
  // convolution
  // input dimension: [minibatch][in_channels][iH][iW]
  // output dimension: [minibatch][out_channels][iH-D(kH-1)][iW-D(kW-1)]

  /* calculate output dimension */
  int outH = inH - d * (kernelH - 1);
  int outW = inW - d * (kernelW - 1);
  float output[kInDim0][out_channel][outH][outW];
  for (int m = 0; m < minibatch; m++) {
    /* set bias */
    for (int i = 0; i < out_channel; i++) {
      for (int h = 0; h < outH; h++) {
        for (int w = 0; w < outW; w++) {
          output[m][i][h][w] = bias[i];
        }
      }
    }
    /* convolution */
    for (int i = 0; i < out_channel; i++) {
      for (int j = 0; j < in_channel; j++) {
        for (int h = 0; h < outH; h++) {
          for (int w = 0; w < outW; w++) {
            for (int p = 0; p < kernelH; p++) {
              for (int q = 0; q < kernelW; q++) {
                output[m][i][h][w] += data[m][j][h+p*d][w+q*d] * weight[i][j][p][q];
              }
            }
          }
        }
      }
    }
  }
}

float* leaky_relu(float* data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = max(data[i], 0) + 0.01 * min(data[i], 0);
  }
  return data;
}

float* relu(float* data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = max(data[i], 0);
  }
  return data;
}

float* softmax_dim1(float* data, int dim0, int dim1) {
  for (int i = 0; i < dim0; i++) {
    float exp_sum = 0;
    for (int j = 0; j < dim1; j++) {
      exp_sum += exp(data[i][j]);
    }
    for (int j = 0; j < dim1; j++) {
      data[i][j] = exp(data[i][j]) / exp_sum;
    }
  }
}

float* sigmoid(float* data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = 1 / (1 + exp(-data[i]));
  }
  return data;
}

float* tanh(float* data, int size) {
  for (int i = 0; i < size; i++) {
    data[i] = tanh(data[i]);
  }
  return data;
}

float* matmul(float* mat1, float* mat2, Dim2 mat1dim, Dim2 mat2dim) {
  if (mat1dim.dim1 != mat2dim.dim0) {
    return nullptr;
  }
  float* output;
  for (int i = 0; i < mat1dim.dim0; i++) {
    for (int k = 0; k < mat2dim.dim1; k++) {
      output[i][k] = 0;
    }
  }
  // matrix multiplication
  for (int i = 0; i < mat1dim.dim0; i++) {
    for (int k = 0; k < mat2dim.dim1; k++) {
      for (int j = 0; j < mat1dim.dim1; j++) {
        output[i][k] += mat1[i][j] * mat2[j][k];
      }
    }
  }
  return output;
}

// element-wise matrix addition
float* elem_add(float* arr1, float* arr2, Dim4 dim) {
  float* output = new float[dim.dim0][dim.dim1][dim.dim2][dim.dim3];
  for (int i = 0; i < dim.dim0; i++) {
    for (int j = 0; j < dim.dim1; j++) {
      for (int k = 0; k < dim.dim2; k++) {
        for (int l = 0; l < dim.dim3; l++) {
          output[i][j][k][l] = arr1[i][j][k][l] + arr2[i][j][k][l];
        }
      }
    }
  }
  return output;
}

// element-wise matrix multiplication
float* elem_mul(float* arr1, float* arr2, Dim4 dim) {
  float* output = new float[dim.dim0][dim.dim1][dim.dim2][dim.dim3];
  for (int i = 0; i < dim.dim0; i++) {
    for (int j = 0; j < dim.dim1; j++) {
      for (int k = 0; k < dim.dim2; k++) {
        for (int l = 0; l < dim.dim3; l++) {
          output[i][j][k][l] = arr1[i][j][k][l] * arr2[i][j][k][l];
        }
      }
    }
  }
  return output;
}

class GraphWaveNet {
 public:
  GraphWaveNet(int num_nodes, float dropout=0.3, float* supports=NULL, bool do_graph_conv=true,
         bool addaptadj=true, float* aptinit=NULL, int in_dim=2, int out_dim=12,
         int residual_channels=32, int dilation_channels=32, bool cat_feat_gc=false,
         int skip_channels=256, int end_channels=512, int kernel_size=2, int blocks=4, int layers=2,
         int apt_size=10);
  ~GraphWaveNet();
  float* forward(float* x);
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
  for (int b = 0; b < m_blocks; b++) {
    int additional_scope = kKernelSize - 1;
    int D = 1;
    for (int i = 0; i < m_layers; i++) {
      // dilated convolutions
      // self.filter_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
      // self.gate_convs.append(Conv1d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
      m_dil_convs[b * m_layers + i] = D;
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

// destructor
GraphWaveNet::~GraphWaveNet() {
  delete[] m_x;
  // todo: delete all other dynamic arrays
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