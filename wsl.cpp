#include<Eigen/Dense>
using namespace Eigen;

ArrayXXf wsl(const ArrayXXf& res)
{
    float  Lambda = 1.0, alpha = 1.2, eps = 0.0001;
    //ArrayXXf res = point_max(array1, array2);
    ArrayXXf l = res.log() + eps;
    int k = res.rows() * res.cols();
    ArrayXXf dy = res.block(1, 0, res.rows() - 1, res.cols()) - res.block(0, 0, res.rows() - 1, res.cols());
    dy = -Lambda / (dy.abs().pow(alpha) + eps);
    ArrayXXf dyy(res.rows(), res.cols());
    dyy << dy, ArrayXXf::Zero(1, res.cols());
    dyy = dyy.reshaped(k, 1);
    ArrayXXf dx = res.block(0, 1, res.rows(), res.cols() - 1) - res.block(0, 0, res.rows(), res.cols() - 1);
    dx = -Lambda / (dx.abs().pow(alpha) + eps);
    ArrayXXf dxx(res.rows(), res.cols());
    dxx << dx, ArrayXXf::Zero(res.rows(), 1);
    dxx = dxx.reshaped(k, 1);
    ArrayXXf b(k, 2);
    b << dxx, dyy;
    ArrayXXf B = b.transpose();
    ArrayXXf A = ArrayXXf::Zero(k, k);
    for (int i = 1; i < k; i++)
    {
        A(i, i - 1) = dyy(i - 1, 0);
    }
    for (int i = res.rows(); i < k; i++) A(i, i - res.rows()) = dxx(i - res.rows(), 0);
    ArrayXXf temp(k + res.rows(), 1);
    temp << ArrayXXf::Zero(res.rows(), 1), dxx;
    ArrayXXf w = temp.block(0, 0, k, 1);
    ArrayXXf e = dxx, s = dyy;
    ArrayXXf temp2(k + 1, 1);
    temp2 << ArrayXXf::Zero(1, 1), dyy;
    ArrayXXf n = temp2.block(0, 0, k, 1);
    ArrayXXf D = 1 - (w + e + n + s);
    ArrayXXf temp3 = ArrayXXf::Zero(k, k);
    for (int i = 0; i < k; i++) temp3(i, i) = D(i, 0);
    A = A + A.transpose() + temp3;
    ArrayXXf a = A.matrix().inverse().array();
    ArrayXXf im = res.transpose().reshaped(k, 1);
    ArrayXXf out = a.matrix() * im.matrix();
    return out.transpose().reshaped(res.rows(), res.cols());

}