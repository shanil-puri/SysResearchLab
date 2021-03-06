// -*- C++ -*-
// SVM with stochastic gradient
// Copyright (C) 2007- Leon Bottou

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA

#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "assert.h"
#include "vectors.h"
#include "gzstream.h"
#include "timer.h"
#include "loss.h"
#include "data.h"

using namespace std;

// ---- Loss function

// Compile with -DLOSS=xxxx to define the loss function.
// Loss functions are defined in file loss.h)
#ifndef LOSS
# define LOSS LogLoss
#endif

// ---- Bias term

// Compile with -DBIAS=[1/0] to enable/disable the bias term.
// Compile with -DREGULARIZED_BIAS=1 to enable regularization on the bias term

#ifndef BIAS
# define BIAS 1
#endif
#ifndef REGULARIZED_BIAS
# define REGULARIZED_BIAS 0
#endif

// ---- Plain stochastic gradient descent

class SvmSgd
{
		public:
				SvmSgd(int dim, double lambda, double eta0  = 0);
				void renorm();
				double wnorm();
				double testOne(const SVector &x, double y, double *ploss, double *pnerr);
				void trainOne(const SVector &x, double y, double eta);
		public:
				void train(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix = "");
				void test(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix = "");
		public:
				double evaluateEta(int imin, int imax, const xvec_t &x, const yvec_t &y, double eta);
				void determineEta0(int imin, int imax, const xvec_t &x, const yvec_t &y);
				//winnie, start
				double get_loss();
				double get_wDivisor();
				double get_wBias();
				void output_w(const char * filename);
				//winnie, end
		private:
				double  lambda;
				double  eta0;
				FVector w;
				double  wDivisor;
				double  wBias;
				double  t;
				double myloss;
};

/// Constructor
		SvmSgd::SvmSgd(int dim, double lambda, double eta0)
: lambda(lambda), eta0(eta0), 
		w(dim), wDivisor(1), wBias(0),
		t(0)
{
}

/// Renormalize the weights
		void
SvmSgd::renorm()
{
		if (wDivisor != 1.0)
		{
				w.scale(1.0 / wDivisor);
				wDivisor = 1.0;
		}
}

/// Compute the norm of the weights
		double
SvmSgd::wnorm()
{
		double norm = dot(w,w) / wDivisor / wDivisor;
		//cout << "\n inside wnorm, norm: \n" << norm << endl;
		//cout << "wDivisor: " << wDivisor << endl;
		//cout << "dot(w,w): " << dot(w,w) << endl;
#if REGULARIZED_BIAS
		norm += wBias * wBias;
#endif
		return norm;
}

/// Compute the output for one example.
		double
SvmSgd::testOne(const SVector &x, double y, double *ploss, double *pnerr)
{
		double s = dot(w,x) / wDivisor + wBias;
		if (ploss)
				*ploss += LOSS::loss(s, y);
		if (pnerr)
				*pnerr += (s * y <= 0) ? 1 : 0;
		return s;
}

/// Perform one iteration of the SGD algorithm with specified gains
		void
SvmSgd::trainOne(const SVector &x, double y, double eta)
{
		double s = dot(w,x) / wDivisor + wBias;
		// update for regularization term
		wDivisor = wDivisor / (1 - eta * lambda);
		if (wDivisor > 1e5) renorm();
		// update for loss term
		double d = LOSS::dloss(s, y);
		if (d != 0)
				w.add(x, eta * d * wDivisor);
		// same for the bias
#if BIAS
		double etab = eta * 0.01;
#if REGULARIZED_BIAS
		wBias *= (1 - etab * lambda);
#endif
		wBias += etab * d;
#endif
}


/// Perform a training epoch
		void 
SvmSgd::train(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
{
		cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
		assert(imin <= imax);
		assert(eta0 > 0);
		for (int i=imin; i<=imax; i++)
		{
				double eta = eta0 / (1 + lambda * eta0 * t);
				trainOne(xp.at(i), yp.at(i), eta);
				t += 1;
		}
		cout << prefix << setprecision(6) << "wNorm=" << wnorm();
#if BIAS
		cout << " wBias=" << wBias;
#endif
		cout << endl;
}

/// Perform a test pass
		void 
SvmSgd::test(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
{
		cout << prefix << "Testing on [" << imin << ", " << imax << "]." << endl;
		assert(imin <= imax);
		double nerr = 0;
		double loss = 0;
		for (int i=imin; i<=imax; i++)
				testOne(xp.at(i), yp.at(i), &loss, &nerr);
		nerr = nerr / (imax - imin + 1);
		loss = loss / (imax - imin + 1);
		double cost = loss + 0.5 * lambda * wnorm();

		//winnie, start
		myloss = loss;
		//winnie, end
		cout << prefix 
				<< "Loss=" << setprecision(12) << loss
				<< " Cost=" << setprecision(12) << cost 
				<< " Misclassification=" << setprecision(4) << 100 * nerr << "%." 
				<< endl;
}

//winnie, start
		double
SvmSgd::get_loss()
{
		return myloss;
}
		double
SvmSgd::get_wDivisor()
{
		return wDivisor;
}
		double
SvmSgd::get_wBias()
{
		return wBias;
}

void SvmSgd::output_w(const char* filename)
{
		//TODO: output parameter w!!!
		VFloat *f1 = w;
		cout << "\n Prepare to output w: \n" ;
		int m = w.size();
		cout << "size of m: " << m << endl;

		ofstream outwfd(filename);
		if(outwfd){
				outwfd << t << endl;
				outwfd << wDivisor << endl;
				outwfd << wBias << endl;
				while(--m >= 0){
						outwfd << *f1++ << "\t" ;
				}
		}
		else{

				cout << t << endl;
				cout << wDivisor << endl;
				cout << wBias << endl;
				while(--m >= 0){
						cout << *f1++ << "\t" ;
				}
		}

}

//winnie, end


/// Perform one epoch with fixed eta and return cost

		double 
SvmSgd::evaluateEta(int imin, int imax, const xvec_t &xp, const yvec_t &yp, double eta)
{
		SvmSgd clone(*this); // take a copy of the current state
		assert(imin <= imax);
		for (int i=imin; i<=imax; i++)
				clone.trainOne(xp.at(i), yp.at(i), eta);
		double loss = 0;
		double cost = 0;
		for (int i=imin; i<=imax; i++)
				clone.testOne(xp.at(i), yp.at(i), &loss, 0);
		loss = loss / (imax - imin + 1);
		cost = loss + 0.5 * lambda * clone.wnorm();
		// cout << "Trying eta=" << eta << " yields cost " << cost << endl;
		return cost;
}

		void 
SvmSgd::determineEta0(int imin, int imax, const xvec_t &xp, const yvec_t &yp)
{
		const double factor = 2.0;
		double loEta = 1;
		double loCost = evaluateEta(imin, imax, xp, yp, loEta);
		double hiEta = loEta * factor;
		double hiCost = evaluateEta(imin, imax, xp, yp, hiEta);
		if (loCost < hiCost)
				while (loCost < hiCost)
				{
						hiEta = loEta;
						hiCost = loCost;
						loEta = hiEta / factor;
						loCost = evaluateEta(imin, imax, xp, yp, loEta);
				}
		else if (hiCost < loCost)
				while (hiCost < loCost)
				{
						loEta = hiEta;
						loCost = hiCost;
						hiEta = loEta * factor;
						hiCost = evaluateEta(imin, imax, xp, yp, hiEta);
				}
		eta0 = loEta;
		cout << "# Using eta0=" << eta0 << endl;
}


// --- Command line arguments

const char *trainfile = 0;
const char *testfile = 0;
bool normalize = true;
double lambda = 1e-5;
int epochs = 5;
int maxtrain = -1;
double err = 0.0513;
const char *outputfile = 0;


		void
usage(const char *progname)
{
		const char *s = ::strchr(progname,'/');
		progname = (s) ? s + 1 : progname;
		cerr << "Usage: " << progname << " [options] trainfile [testfile]" << endl
				<< "Options:" << endl;
#define NAM(n) "    " << setw(16) << left << n << setw(0) << ": "
#define DEF(v) " (default: " << v << ".)"
		cerr << NAM("-lambda x")
				<< "Regularization parameter" << DEF(lambda) << endl
				<< NAM("-epochs n")
				<< "Number of training epochs" << DEF(epochs) << endl
				<< NAM("-dontnormalize")
				<< "Do not normalize the L2 norm of patterns." << endl
				<< NAM("-maxtrain n")
				<< "Restrict training set to n examples." << endl
				<< NAM("-error x")
				<< "Convege error rate." << endl
				<< NAM("-output filename")
				<< "The file that store wDivisor and wBias." << endl;
#undef NAM
#undef DEF
		::exit(10);
}

		void
parse(int argc, const char **argv)
{
		cout << "test1";
		for (int i=1; i<argc; i++)
		{
				const char *arg = argv[i];
				if (arg[0] != '-')
				{
						cout << "test2";
						if (trainfile == 0){
								trainfile = arg;
								cout << "trainfile: " << trainfile << endl;
						}
						else if (testfile == 0){
								testfile = arg;
								cout << "testfile: " << testfile << endl;
						}
						else{
								cout << "else\n";
								usage(argv[0]);
						}
				}
				else
				{
						cout << "test3";
						while (arg[0] == '-') 
								arg += 1;
						string opt = arg;
						if (opt == "lambda" && i+1<argc)
						{
								lambda = atof(argv[++i]);
								assert(lambda>0 && lambda<1e4);
						}
						else if (opt == "epochs" && i+1<argc)
						{
								epochs = atoi(argv[++i]);
								assert(epochs>0 && epochs<1e6);
						}
						else if (opt == "dontnormalize")
						{
								normalize = false;
						}
						else if (opt == "maxtrain" && i+1 < argc)
						{
								maxtrain = atoi(argv[++i]);
								assert(maxtrain > 0);
						}
						//winnie, start
						else if (opt == "error" && i+1<argc)
						{
								cout << "test4";
								err = atof(argv[++i]);

								cout << "err: " << err << endl;
								assert(err>0 && err<1);
						}
						else if (opt == "output" && i+1<argc)
						{
								cout << "test5";
								outputfile = argv[++i];

						}	//winnie, end
						else
						{
								cout << "test5";
								cerr << "Option " << argv[i] << " not recognized." << endl;
								usage(argv[0]);
						}

				}
		}
		if (! trainfile)
		{
				usage(argv[0]);
				cout << "test6";
		}
}

		void 
config(const char *progname)
{
		cout << "# Running: " << progname;
		cout << " -lambda " << lambda;
		cout << " -epochs " << epochs;
		cout << " -error " << err;
		//cout << " -output " << outputfile;
		if (! normalize) cout << " -dontnormalize";
		if (maxtrain > 0) cout << " -maxtrain " << maxtrain;
		cout << endl;
#define NAME(x) #x
#define NAME2(x) NAME(x)
		cout << "# Compiled with: "
				<< " -DLOSS=" << NAME2(LOSS)
				<< " -DBIAS=" << BIAS
				<< " -DREGULARIZED_BIAS=" << REGULARIZED_BIAS
				<< endl;
}

// --- main function

int dims;
xvec_t xtrain;
yvec_t ytrain;
xvec_t xtest;
yvec_t ytest;

int main(int argc, const char **argv)
{
		parse(argc, argv);
		config(argv[0]);
		if (trainfile)
				load_datafile(trainfile, xtrain, ytrain, dims, normalize, maxtrain);
		if (testfile)
				load_datafile(testfile, xtest, ytest, dims, normalize);
		cout << "# Number of features " << dims << "." << endl;
		// prepare svm
		int imin = 0;
		int imax = xtrain.size() - 1;
		int tmin = 0;
		int tmax = xtest.size() - 1;
		SvmSgd svm(dims, lambda);
		Timer timer;
		Timer totalTimer;
		// determine eta0 using sample
		int smin = 0;
		int smax = imin + min(1000, imax);
		totalTimer.start();
		timer.start();
		svm.determineEta0(smin, smax, xtrain, ytrain);
		timer.stop();
		// train

		//winnie, modify code end condition from number of epochs to misclassification rate
		//  for(int i=0; i<epochs; i++)
		double loss_new = 1, loss_old = 0;
		//svm.test(imin, imax, xtrain, ytrain);
		//loss_new = svm.get_loss();

		int i = 0;
		while(loss_old - loss_new > 1e-5 || loss_new - loss_old > 1e-5)
		{

				cout << "--------- Epoch " << i++ << "." << endl;
				//cout << "wDivisor: " << svm.get_wDivisor() << endl;
				//cout << "wBias: " << svm.get_wBias() << endl;
				//				timer.start();
				svm.train(imin, imax, xtrain, ytrain);
				//				timer.stop();
				//				cout << "Total training time in this epoch " << setprecision(6) 
				//						<< timer.elapsed() << " secs." << endl;
				svm.test(imin, imax, xtrain, ytrain, "train: ");
				loss_old = loss_new;
				loss_new = svm.get_loss();

				if (tmax >= tmin)
						svm.test(tmin, tmax, xtest, ytest, "test:  ");

		}

		totalTimer.stop();

		cout << "Total time: " << setprecision(6) 
				<< totalTimer.elapsed() << " secs." << endl;
		//winnie, output parameter wDivisor and wBias:
		//		ofstream outputf(outputfile);
		//		if(outputf){
		//				//cout << "wDivisor: " << svm.get_wDivisor() << endl;
		//				//cout << "wBias: " << svm.get_wBias() << endl;
		//				outputf << "wDivisor: " << svm.get_wDivisor() << endl;
		//				outputf << "wBias: " << svm.get_wBias() << endl;
		//
		//		}
		//
		//		outputf.close();


		//		const char * outputfile = 0;
		svm.output_w(outputfile);

		return 0;
}
