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
#include "src/sampling.hpp"
#include <sys/time.h>
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
				void pre_trainOne(const SVector &x, double y, double eta);
		public:
				void train(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix = "");
				void pre_train(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix = "");
				double test(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix = "");
				//void test(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix = "");
		public:
				double evaluateEta(int imin, int imax, const xvec_t &x, const yvec_t &y, double eta);
				double my_evaluateEta(int imin, int imax, const xvec_t &x, const yvec_t &y, double eta);
				double my_evaluateEta_v1(int imin, int imax, const xvec_t &x, const yvec_t &y, double eta);
				void determineEta0(int imin, int imax, const xvec_t &x, const yvec_t &y);
				void my_determineEta0(int imin, int imax, const xvec_t &x, const yvec_t &y, double & t);
				void my_determineEta0_v2(int imin, int imax, const xvec_t &x, const yvec_t &y, double & t, double hiEta);
				void my_determineEta0_v3(int imin, int imax, const xvec_t &x, const yvec_t &y);
				double compareEta0(int imin, int imax, const xvec_t &xp, const yvec_t &yp, double midEta, double length, int count);
				//winnie, start
				double get_cost();
				double get_wDivisor();
				double get_wBias();
				void set_wDivisor(double value);
				void set_wBias(double value);

				void output_w(const char * filename);
				void init_w(const char * filename, int m);
				void get_w(FVector& w2);
				void get_delta_w(FVector w2, map <double, int> & delta_w);

				//temp move it from private to public
				double  t;  
				double  eta0;
				//winnie, end of temp move

				//private:
		public:
				double  lambda;
				FVector w;
				double  wDivisor;
				double  wBias;
				double mycost;
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
		//		cout << "\n inside wnorm, norm: \n" << norm << endl;
		//		cout << "wDivisor: " << wDivisor << endl;
		//		cout << "dot(w,w): " << dot(w,w) << endl;
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
		//cout << "[testOne: *ploss: ]" << *ploss << endl;
		return s;
}

/// Perform one iteration of the SGD algorithm with specified gains
		void
SvmSgd::trainOne(const SVector &x, double y, double eta)
{
		double s = dot(w,x) / wDivisor + wBias;
		// update for regularization term
		wDivisor = wDivisor / (1 - eta * lambda);
		//cout << "[trainOne:] wDivisor: " << wDivisor << endl;
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
//Winnie, train on samples, because the data has been randomly shuffled, just use the first m instances.
		void
SvmSgd::pre_trainOne(const SVector &x, double y, double eta)
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
SvmSgd::pre_train(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
{
		cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
		assert(imin <= imax);
		assert(eta0 > 0);

		int my_max = (imin + 500 > imax? imax : imin + 500);
		//cout << "wDivisor: " << wDivisor << "  wBias: " << wBias<< endl;
		for (int i=imin; i<= my_max; i++)
		{
				double eta = eta0 / (1 + lambda * eta0 * t);
				trainOne(xp.at(i), yp.at(i), eta);
				t += 1;
		}

		//cout << "\nAfter training: \n  wDivisor: " << wDivisor << "  wBias: " << wBias<< endl;
		cout << prefix << setprecision(6) << "wNorm=" << wnorm();
#if BIAS
		cout << " wBias=" << wBias;
#endif
		cout << endl;
}


/// Perform a training epoch
		void 
SvmSgd::train(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
{
		cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
		assert(imin <= imax);
		assert(eta0 > 0);

		//cout << "wDivisor: " << wDivisor << "  wBias: " << wBias<< endl;
		for (int i=imin; i<=imax; i++)
		{
				double eta = eta0 / (1 + lambda * eta0 * t);
				//cout << "[my_evaluateEta:] Eta: " << eta << endl;
				trainOne(xp.at(i), yp.at(i), eta);
				t += 1;
		}

		//cout << "\nAfter training: \n  wDivisor: " << wDivisor << "  wBias: " << wBias<< endl;
		cout << prefix << setprecision(6) << "wNorm=" << wnorm();
#if BIAS
		cout << " wBias=" << wBias;
#endif
		cout << endl;
}

/// Perform a test pass
		double	
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
		mycost = cost;
		//winnie, end
		cout << prefix 
				<< "Loss=" << setprecision(12) << loss
				<< " Cost=" << setprecision(12) << cost 
				<< " Misclassification=" << setprecision(4) << 100 * nerr << "%." 
				<< endl;
		return cost;
}

//winnie, start
		double
SvmSgd::get_cost()
{
		return mycost;
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
		void
SvmSgd::get_w(FVector& w2)
{
		w2 = w;
}
		void
SvmSgd::get_delta_w(FVector w2, map <double, int> & delta_w)
{
		double value;
		cout << "w2.size(): " << w2.size() << endl;
		cout << "w2[0] value: " << w2[0] << endl;
		for(int i = 1; i < w2.size(); i++){
				value = fabs(w2[i] - w[i]);
				delta_w.insert(pair<double, int>(value, i - 1));
		}
}


void SvmSgd::set_wDivisor(double value)
{
		wDivisor = value;
}
void SvmSgd::set_wBias(double value)
{
		wBias = value;
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
void SvmSgd::init_w(const char* filename, int m)
{
		//TODO: output parameter w!!!
		VFloat *f1 = w;
		ifstream inwfd(filename);

		if(inwfd){
				inwfd >> t;   //use t and eta0 to calc eta
				inwfd >> wDivisor;
				inwfd >> wBias;
				while(--m >= 0){
						inwfd >> *f1++;
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



		double 
SvmSgd::my_evaluateEta_v1(int imin, int imax, const xvec_t &xp, const yvec_t &yp, double eta00)
{
		SvmSgd clone(*this); // take a copy of the current state

		cout << "[my_evaluateEta: clone.wDivisor: ]" << setprecision(12) << clone.wDivisor << " clone.t: " << clone.t << " clone.eta0: " << clone.eta0 << endl; 
		cout << "Trying eta=" << eta00 ;

		assert(imin <= imax);

		clone.eta0 = eta00;
		clone.t = 0;

		clone.train(imin, imax, xp, yp);
		return clone.test(imin, imax, xp, yp, "train");

		// cout << "Trying eta=" << eta << " yields cost " << cost << endl;
}
//winnie, end, my_evaluateEta


//winnie, start, my_evaluateEta, do not use fixed eta

		double 
SvmSgd::my_evaluateEta(int imin, int imax, const xvec_t &xp, const yvec_t &yp, double eta00)
{
		SvmSgd clone(*this); // take a copy of the current state

		cout << "[my_evaluateEta: clone.wDivisor: ]" << setprecision(12) << clone.wDivisor << " clone.t: " << clone.t << " clone.eta0: " << clone.eta0 << endl; 
		cout << "Trying eta=" << eta00 ;

		assert(imin <= imax);
		double _t = 0;
		double eta = 0;
		for (int i=imin; i<=imax; i++){
				eta = eta00 / (1 + lambda * eta00 * _t);
				//cout << "[my_evaluateEta:] Eta: " << eta << endl;
				clone.trainOne(xp.at(i), yp.at(i), eta);
				_t++;
		}
		double loss = 0;
		double cost = 0;
		for (int i=imin; i<=imax; i++)
				clone.testOne(xp.at(i), yp.at(i), &loss, 0);
		loss = loss / (imax - imin + 1);
		cost = loss + 0.5 * lambda * clone.wnorm();
		cout <<" yields loss " << loss << endl;
		// cout << "Trying eta=" << eta << " yields cost " << cost << endl;
		return cost;
}
//winnie, end, my_evaluateEta
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

//winnie, calibrate determineEta0 for our method, enlarge the range of checked eta0 value
		void 
SvmSgd::my_determineEta0(int imin, int imax, const xvec_t &xp, const yvec_t &yp, double & t)
{
		const double factor = 2;
		double loEta = 2;
		double loCost = evaluateEta(imin, imax, xp, yp, loEta);
		double hiEta = loEta + factor; //enlarge the range of checked eta0
		double hiCost = evaluateEta(imin, imax, xp, yp, hiEta);
		double eta = 0;
		//		t = 1;

		cout << "imin: " << imin << " imax: " << imax << endl;
		double temp_cost = evaluateEta(imin, imax, xp, yp, 0.0625);
		cout << "temp_cost of eta0=0.0625: " << temp_cost << endl; 

		if (loCost < hiCost){
				while (loCost < hiCost)
				{
						hiEta = loEta;
						hiCost = loCost;

						cout << "loCost0: " << loCost << " loEta: " << loEta << endl;
						//						loEta = hiEta / factor;
						loEta = loEta / (1 + lambda * loEta * t);
						loCost = evaluateEta(imin, imax, xp, yp, loEta);
						cout << "loCost1: " << loCost << " loEta: " << loEta << " t: " << t<< endl;
						t += 100000;

				}

				loCost = evaluateEta(imin, imax, xp, yp, loEta / 2*factor);
				cout << "loCost2: " << loCost << " loEta: " << loEta / 2*factor << endl;
				//		while(loCost < hiCost)
				//		{
				//				hiEta = loEta;
				//				hiCost = loCost;

				//				cout << "loCost2: " << loCost << " loEta: " << loEta << endl;
				//				loEta = hiEta * factor;
				//				loCost = evaluateEta(imin, imax, xp, yp, loEta);
				//				cout << "loCost3: " << loCost << " loEta: " << loEta << endl;

				//		}


				eta0 = hiEta;
		}
		else if (hiCost < loCost){
				while (hiCost < loCost)
				{
						loEta = hiEta;
						loCost = hiCost;
						hiEta = hiEta / (1 + lambda * hiEta * t);
						hiCost = evaluateEta(imin, imax, xp, yp, hiEta);
						//hiCost = evaluateEta(imin, imax, xp, yp, hiEta);
						cout << "hiCost: " << hiCost << endl;
						cout << "LoCost: " << loCost << endl;
						t += 100000;
				}
				eta0 = loEta;

		}
		t -= 2*100000;
		cout << "# Using eta0=" << eta0 << endl;
}


//winnie, end my_determineEta0
//winnie, start of second version of my_determinEta0
		void 
SvmSgd::my_determineEta0_v2(int imin, int imax, const xvec_t &xp, const yvec_t &yp, double & t, double hiEta)
{
		const double factor = 2;
		//		double hiEta = 4;
		double loEta = hiEta / (1 + lambda * hiEta * t);
		double loCost = evaluateEta(imin, imax, xp, yp, loEta);
		double hiCost = evaluateEta(imin, imax, xp, yp, hiEta);
		double eta = 0;

		cout << "imin: " << imin << " imax: " << imax << endl;
		cout << "loCost: " << loCost << " hiCost: " << hiCost << endl;

		if (loCost < hiCost){
				while (loCost < hiCost)
				{
						//hiEta = loEta;
						hiCost = loCost;

						cout << "loCost0: " << loCost << " loEta: " << loEta << endl;
						eta = loEta / (1 + lambda * loEta * t);
						loCost = evaluateEta(imin, imax, xp, yp, eta);
						cout << "loCost1: " << loCost << " loEta: " << eta << " t: " << t<< endl;
						t += 100000;

				}

				loCost = evaluateEta(imin, imax, xp, yp, loEta / 2*factor);
				cout << "loCost2: " << loCost << " loEta: " << loEta / 2*factor << endl;
				eta0 = hiEta;
		}
		else if (hiCost < loCost){
				hiEta *= factor;
				my_determineEta0_v2(imin, imax, xp, yp, t, hiEta);
		}
		t -= 2*100000;
		cout << "# Using eta0=" << eta0 << endl;
}

//winnie, end of second version of my_determinEta0

//winnie, start third version of my_determineEta0
double
SvmSgd::compareEta0(int imin, int imax, const xvec_t &xp, const yvec_t &yp, double midEta, double length, int count){

		cout << "Count: " << count << endl;
		count++;
		double loEta = midEta - length;
		double hiEta = midEta + length;

		cout << "HiEta: " <<setprecision(12)  << hiEta << " midEta: " << midEta << " loEta: " << loEta << endl;
		double loCost = my_evaluateEta(imin, imax, xp, yp, loEta);
		double hiCost = my_evaluateEta(imin, imax, xp, yp, hiEta);
		double midCost =my_evaluateEta(imin, imax, xp, yp, midEta);
		cout << "This round \nHiCost: " << hiCost << " midCost: " << midCost << " loCost: " << loCost << endl;
		if(midEta == 0)
				compareEta0(imin, imax, xp, yp, length/2, length/2,count);
		else if(loEta <0 ){
				midEta = (midEta + length) / 2;
				length = midEta;
				compareEta0(imin, imax, xp, yp, midEta, length, count);
		} 
		//else if(abs(loCost - midCost) == 0 && midCost <= hiCost)
		else if(abs(loCost - midCost) < 1e-5 && midCost <= hiCost){
				if(loCost < midCost)
						return loEta;
				else
						return midEta;

		}
		//else if(abs(midCost - hiCost) == 0 && loCost >= midCost)
		else if(abs(midCost - hiCost) < 1e-5 && loCost >= midCost){
				if(midCost < hiCost)
						return midEta;
				else
						return hiEta;

		}
		else if(loCost < midCost){
				midEta = loEta; 

				cout << "#loCost < midCost\n";
				//cout << "HiCost: " << hiCost << " midCost: " << midCost << " loCost: " << loCost << endl;
				//cout << "HiEta: " <<setprecision(12)<< hiEta << " midEta: " << midEta << " loEta: " << loEta << endl;

				compareEta0(imin, imax, xp, yp, midEta, length, count);
		}
		else if(midCost < hiCost){
				length = length / 2;

				cout << "# midCost < hiCost\n";
				//cout << "HiCost: " << hiCost << " midCost: " << midCost << " loCost: " << loCost << endl;
				//cout << "HiEta: " <<setprecision(12)  << hiEta << " midEta: " << midEta << " loEta: " << loEta << endl;
				compareEta0(imin, imax, xp, yp, midEta, length, count);
		}
		else if(midCost > hiCost){
				midEta = hiEta;

				cout << "# midCost > hiCost\n\n";
				compareEta0(imin, imax, xp, yp, midEta, length, count);


		}
		else{
				cout << "Wrong branch!\n" ;
				cout << "HiCost: " << hiCost << " midCost: " << midCost << " loCost: " << loCost << endl;
				cout << "HiEta: " <<setprecision(12)  << hiEta << " midEta: " << midEta << " loEta: " << loEta << endl;
				return -1;
		}

}


		void 
SvmSgd::my_determineEta0_v3(int imin, int imax, const xvec_t &xp, const yvec_t &yp)
{
		cout << "Number of smaples: " << imax - imin << endl;
		double loEta = 2;
		double length = 2;
		double hiEta = loEta + length;
		double loCost = my_evaluateEta(imin, imax, xp, yp, loEta);
		double hiCost = my_evaluateEta(imin, imax, xp, yp, hiEta);

		double midEta = (hiCost > loCost) ? hiEta : loEta; 
		eta0 = compareEta0(imin, imax, xp, yp, midEta, length, 1);

		cout << "# Using eta0=" << eta0 << endl;
}




//winnie, end of third version of my_determineEta0



// --- Command line arguments

const char *trainfile = 0;
const char *testfile = 0;
bool normalize = true;
double lambda = 1e-5;
int epochs = 5;
int maxtrain = -1;
double err = 0.0513;
const char *outputfile = 0;
const char *database_file = 0;
const char *database_dir = 0;
char distancetype = 0;
double revalue = 0;
const char *sample_file = 0;
const char *initpara_file = 0;


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
				<< NAM("-samplefile filename")
				<< "filename used to calculate pdf" << endl
				<< NAM("-database directory")
				<< "filename used to calculate pdf" << endl
				<< NAM("-databasedir directory")
				<< "directory that contains all the memorized initial parameter files" << endl
				<< NAM("-initpara filename")
				<< "file contains the initial parameters want to use, for testing purpose" << endl;
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
								//cout << "test4";
								err = atof(argv[++i]);

								//cout << "err: " << err << endl;
								assert(err>0 && err<1);
						}
						else if (opt == "output" && i+1<argc)
						{
								//cout << "test5";
								outputfile = argv[++i];

						}	
						else if (opt == "samplefile" && i+1<argc) //file for sampling and pdf calculation
						{
								//cout << "test9";
								sample_file = argv[++i];

						}
						else if (opt == "database" && i+1<argc) // directory of dens file
						{
								//cout << "test6";
								database_file = argv[++i];

						}
						else if (opt == "initpara" && i+1<argc) //file with initial parameters,!!cannot be used with samplefile option!!
						{
								//cout << "test10";
								initpara_file = argv[++i];
								cout << "initpara_file " << initpara_file << endl;

						}	
						else if (opt == "databasedir" && i+1<argc) // directory of initial parameter files,!!used with samplefile option!!
						{
								//cout << "test7";
								database_dir = argv[++i];

						}	
						else if (opt == "type" && i+1<argc) // directory of initial parameter files,!!used with samplefile option!!
						{
								//cout << "test7";
								distancetype = *argv[++i];

						}
						else if (opt == "costbound" && i+1<argc) // when test cost < costbound, return the whole program
						{
								//cout << "test7";
								const char * re_str = argv[++i];
								revalue = atof(re_str);

						}
						//winnie, end
						else
						{
								cout << "test8";
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

//winnie, add one function to calculate predicted converge iterations for dimension reduction
double predict_iter(double delta_1, double delta_2, double l1){
		double d = delta_1 - delta_2;
		if(d < 0)
				d = -d;

		double l2 = l1, delta_3 = delta_2 - d, z = 0;
		while(delta_3 > 0){
				l2 -= delta_3;
				delta_3 -= d;

				z += 1;
		}
		delta_3 += d;
		z += l2 / delta_3;
		cout << "z: " << z << " l2: " << l2 << " delta_3: " << delta_3 << endl;
		return z;
}

//winnie, iteration for dimension reduction end 

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
		// determine eta0 using sample
		int smin = 0;
		int smax = imin + min(1000, imax);
		// train

		Timer totalTimer, overheadtimer, exetimer;

		totalTimer.start();
		//winnie, initial wDivisor and wBias

		timeval t1, t4, t5, t6, t7, t8;
		overheadtimer.start();
		if(sample_file){
				int sample_size = 500;
				int bin_num = 20;
				int num_compare = 49;
				int dimension = dims - 1; //The first feature is the classification, does not count in sampling.


				gettimeofday(&t1, NULL);

				Sampling<double> selector(imax, 0 ,
								dimension, sample_size, bin_num, num_compare);

				selector.do_sampling(sample_file);
				gettimeofday(&t4, NULL);
				selector.calc_ecdf();		
				gettimeofday(&t5, NULL);
				//	std::cout << "test init kmeans " << std::endl;
				//step3a, do database search
				if(num_compare <= 0){
						cerr << "Number of comparison is less than or equal to 0" << endl;
						return -1;
				}
				else{
						//TODO: make the comparison choose the second best result, when we use data base that contains dataset itself

						//winnie, prerun several iterations, and log some information
						int prerun_iters = 3;
						FVector old_w(dimension);

						svm.determineEta0(smin, smax, xtrain, ytrain);

						for(int i = 0; i < prerun_iters; i ++ ){

								svm.get_w(old_w);
								svm.train(imin, imax, xtrain, ytrain);
								//svm.test(imin, imax, xtrain, ytrain, "train: ");

						}
						//get the idx of dimensions	

						map <double, int> delta_w;
						svm.get_delta_w(old_w, delta_w);
						//int reducedDimNum[8] = {1, 2, 3, 4, 8, 16, 32, 50};
						int reducedDimNum[8] = {1, 3, 8, 16, 32, 50, 64, 128};
						int selected_id[9];
						multimap <double, int> error_dim;  //error value and dimension
						for(int iout = 0; iout < 8; iout++){

								vector<int> reducedDimIdx(reducedDimNum[iout]);
								map<double, int>::reverse_iterator rmit = delta_w.rbegin();
								double sum_value = 0;
								for(int i = 0; i < reducedDimNum[iout]; i ++){
										reducedDimIdx[i] = rmit->second;
										sum_value += rmit->first;
										//cout << "reduceDim: " << reducedDimIdx[i] << " value " << rmit->first << endl;
										rmit++;
								}
								cout << "sum_value: " << sum_value << " with dim num: " << reducedDimNum[iout] << endl;


								//cout << "distancetype: " << distancetype << endl;
								selected_id[iout] = selector.search_database('b', database_file, distancetype, reducedDimIdx);

								gettimeofday(&t6, NULL);
								if(selected_id[iout] < 0)
										return -1;
								else{
										std::cout << "selected id: " << selected_id[iout] << std::endl;
										//Do data customization, use norm.cpp program, 
										stringstream ss;//create a stringstream
										ss << setfill('0') << setw(2) << selected_id[iout];
										string filename = string(database_dir) + "/" + ss.str() + ".txt";
										//						double value1, value2;
										//						selector.load_memo(filename.c_str(), &value1, &value2);
										cout << "Init with file: " << filename << endl;
										svm.init_w(filename.c_str(), dims);
										gettimeofday(&t7, NULL);

										//run one iter use current w

										double pre_loss_1; //, pre_loss_2;
										//double pre_loss_delta_1; //, pre_loss_delta_2; 
										//svm.train(imin, imax, xtrain, ytrain);
										svm.test(tmin, tmax, xtest, ytest, "test:  ");
										pre_loss_1 = svm.get_cost();
										/*
										   svm.train(imin, imax, xtrain, ytrain);
										   svm.test(tmin, tmax, xtest, ytest, "test:  ");
										   pre_loss_2 = svm.get_cost();

										   pre_loss_delta_1 = pre_loss_1 - pre_loss_2; */
										double pre_loss_delta_ratio = pre_loss_1;  //method A
										//double pre_loss_delta_ratio = pre_loss_delta_1; //method B
										//double pre_loss_delta_ratio = -svm.t; //method C
										cout << "-svm.t: " << -svm.t << endl; 

										//svm.train(imin, imax, xtrain, ytrain);
										//svm.test(tmin, tmax, xtest, ytest, "test:  ");
										//pre_loss_3 = svm.get_cost();
										//pre_loss_delta_2 = pre_loss_2 - pre_loss_3;

										/*
										   double pre_loss_delta_ratio;
										   if(pre_loss_delta_1 < 0 || pre_loss_delta_2 < 0){
										   pre_loss_delta_ratio = 1000;
										   }
										   else
										   pre_loss_delta_ratio = predict_iter(pre_loss_delta_1, pre_loss_delta_2, pre_loss_2);
										   cout << "pre_loss_1: " << pre_loss_1 << endl;
										   cout << "pre_loss_2: " << pre_loss_2 << endl;
										   cout << "pre_loss_3: " << pre_loss_3 << endl;
										   cout << "pre_loss_delta_1: " << pre_loss_delta_1 << endl;
										   cout << "pre_loss_delta_2: " << pre_loss_delta_2 << endl;
										   */
										cout << "insert pre_loss_ratio " << pre_loss_delta_ratio << " at dim " << reducedDimNum[iout] << endl;
										error_dim.insert(pair<double, int>(pre_loss_delta_ratio, iout));
										//error_dim.insert(pair<double, int>(fabs(pre_loss_new - pre_loss_old), iout));


								}



						}
						stringstream ss;//create a stringstream
						ss << setfill('0') << setw(2) << selected_id[error_dim.begin()->second];
						string filename = string(database_dir) + "/" + ss.str() + ".txt";
						cout << "Init with: " << filename << endl;

						svm.init_w(filename.c_str(), dims);

						cout << "chosen dim num: " << reducedDimNum[error_dim.begin()->second] << " id: " << selected_id[error_dim.begin()->second] << endl;
						cout << "error of each dim_reduction: " << endl;
						cout << error_dim.size() << "in total\n" ;

						for(map<double, int>::iterator mit = error_dim.begin(); mit != error_dim.end(); mit++){
								cout << mit->second << " " << mit->first << endl;
						}

						//winnie, end of prerun


				}



		}
		else{
				if(!initpara_file){
						cerr << "initpara_file does not exist\n";	
						
				}
				else
						svm.init_w(initpara_file, dims);
		}

		overheadtimer.stop();

		//winnie, end of initial wDivisor and wBias
		exetimer.start();
		timer.start();
		//double temp_t = 1;

		//Original version
		//svm.determineEta0(imin, imax, xtrain, ytrain);
		//svm.determineEta0(smin, smax, xtrain, ytrain);
		//svm.t = 0;


		//Version 3 --------------------------------------------------------
		svm.my_determineEta0_v3(imin, imax, xtrain, ytrain);
		//svm.my_determineEta0_v3(smin, smax, xtrain, ytrain);
		//svm.t = 0;
		//svm.t = 500000;

		//Version 2 --------------------------------------------------------
		//svm.my_determineEta0_v2(smin, smax, xtrain, ytrain, svm.t, 4);
		//svm.my_determineEta0_v2(imin, imax, xtrain, ytrain, svm.t, 4);


		//svm.my_determineEta0(imin, imax, xtrain, ytrain, svm.t);

		timer.stop();


		//svm.eta0 = svm.eta0 / 10;
		//svm.t = 0;
		cout << "New eta0: " << svm.eta0 << " t: " << svm.t << endl;
		cout << "eta: " << setprecision(12) << svm.eta0 / (1 + lambda * svm.eta0 * svm.t) << endl;

		//winnie, modify code end condition from number of epochs to misclassification rate
		double loss_new = 1, loss_old = 1;

		int i = 0;
		//  for(int i=0; i<epochs; i++)
		if(revalue == 0){
				svm.test(imin, imax, xtrain, ytrain);
				loss_new = svm.get_cost();

				while(loss_old - loss_new > 1e-5 || loss_new - loss_old > 1e-5)
				{

						cout << "--------- Epoch " << ++i << endl;

						//timer.start();
						svm.train(imin, imax, xtrain, ytrain);
						//timer.stop();
						//cout << "Total training time " << setprecision(6) 
						//		<< timer.elapsed() << " secs." << endl;
						svm.test(imin, imax, xtrain, ytrain, "train: ");

						if (tmax >= tmin)
								svm.test(tmin, tmax, xtest, ytest, "test:  ");

						loss_old = loss_new;
						loss_new = svm.get_cost();
				}	

		}
		else{
				cout<< "[Main: svm.wDivisor: ]" << setprecision(12) <<svm.wDivisor << " svm.t: " << svm.t << "svm.eta0: " << svm.eta0 << endl;
				while(loss_new > revalue)
				{

						cout << "--------- Epoch " << ++i << endl;

						//timer.start();
						svm.train(imin, imax, xtrain, ytrain);
						//timer.stop();
						//cout << "Total training time " << setprecision(6) 
						//		<< timer.elapsed() << " secs." << endl;
						svm.test(imin, imax, xtrain, ytrain, "train: ");

						loss_old = loss_new;
						loss_new = svm.get_cost();
						if (tmax >= tmin)
								svm.test(tmin, tmax, xtest, ytest, "test:  ");

						if(i >= 200){
								cout << "Exit because reach 2 epochs\n";	
								break;
						}
				}

		}

		cout << "Loss_new: " << loss_new << " Loss_old:" << loss_old << endl;

		exetimer.stop();
		totalTimer.stop();

		cout << "Total time: " << setprecision(6)
				<< totalTimer.elapsed() << " secs." << endl;

		cout << "Total exe time: " << setprecision(6)
				<< exetimer.elapsed() << " secs." << endl;

		cout << "Total overhead time: " << setprecision(6)
				<< overheadtimer.elapsed() << " secs." << endl;
		if(sample_file){


				double elapsedTime;
				elapsedTime =(t4.tv_sec - t1.tv_sec) * 1000.0 + (t4.tv_usec - t1.tv_usec) / 1000.0;   // sec and us to ms
				std::cout <<"Overhead s1: "<< elapsedTime << " ms.\n";

				elapsedTime =(t5.tv_sec - t4.tv_sec) * 1000.0 + (t5.tv_usec - t4.tv_usec) / 1000.0;   // sec and us to ms
				std::cout <<"Overhead s2: "<< elapsedTime << " ms.\n";

				elapsedTime =(t6.tv_sec - t5.tv_sec) * 1000.0 + (t6.tv_usec - t5.tv_usec) / 1000.0;   // sec and us to ms
				std::cout <<"Overhead s3: "<< elapsedTime << " ms.\n";

				elapsedTime =(t7.tv_sec - t6.tv_sec) * 1000.0 + (t7.tv_usec - t6.tv_usec) / 1000.0;   // sec and us to ms
				std::cout <<"Overhead s4: "<< elapsedTime << " ms.\n";

				elapsedTime =(t8.tv_sec - t7.tv_sec) * 1000.0 + (t8.tv_usec - t7.tv_usec) / 1000.0;   // sec and us to ms
				std::cout <<"Overhead s5: "<< elapsedTime << " ms.\n";


		}
		//winnie, output parameter wDivisor and wBias:
		cout << "wDivisor: " << svm.get_wDivisor() << endl;
		cout << "wBias: " << svm.get_wBias() << endl;

		svm.output_w(outputfile);

		return 0;
}
