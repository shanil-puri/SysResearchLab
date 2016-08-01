#ifndef KMEANS_HPP
#define KMEANS_HPP

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
 
typedef struct { double x, y, z; int group; } point_t, *point;
 
double randf(double m)
{
	return m * rand() / (RAND_MAX - 1.);
}
 
point gen_xy(int count, double radius)
{
	double ang, r;
	point p, pt = (point_t *) malloc(sizeof(point_t) * count);
 
	/* note: this is not a uniform 2-d distribution */
	for (p = pt + count; p-- > pt;) {
		ang = randf(2 * M_PI);
		r = randf(radius);
		p->x = r * cos(ang);
		p->y = r * sin(ang);
	}
 
	return pt;
}
 
inline double dist2(point a, point b)
{
	double x = a->x - b->x, y = a->y - b->y, z = a->z - b->z;
	return x*x + y*y + z*z;
}
 
inline int
nearest(point pt, point cent, int n_cluster, double *d2)
{
	int i, min_i;
	point c;
	double d, min_d;
#	define for_n for (c = cent, i = 0; i < n_cluster; i++, c++)
	for_n {
		min_d = HUGE_VAL;
		min_i = pt->group;
		for_n {
			std::cout << c->x << " : " << c->y << " : " << c->z << " : dist : " << dist2(c,pt) <<std::endl;
			if (min_d > (d = dist2(c, pt))) {
				min_d = d; min_i = i;
			}
		}
	}
	if (d2) *d2 = min_d;
	return min_i;
}
 
void kpp(point pts, int len, point cent, int n_cent)
{
#	define for_len for (j = 0, p = pts; j < len; j++, p++)
	int i, j;
	int n_cluster;
	double sum, *d = (double *) malloc(sizeof(double) * len);
 
	point p, c;
	cent[0] = pts[ rand() % len ];
	for (n_cluster = 1; n_cluster < n_cent; n_cluster++) {
		sum = 0;
		for_len {
			nearest(p, cent, n_cluster, d + j);
			sum += d[j];
		}
		sum = randf(sum);
		for_len {
			if ((sum -= d[j]) > 0) continue;
			cent[n_cluster] = pts[j];
			break;
		}
	}
	for_len p->group = nearest(p, cent, n_cluster, 0);
	free(d);
}
 
point lloyd(point pts, int len, cv::Mat cents, int n_cluster)
{
	int i, j, min_i, clust;
	int changed;
 
	point cent = (point_t *) malloc(sizeof(point_t) * n_cluster), p, c;
 
	/* assign init grouping randomly */
	//for_len p->group = j % n_cluster;
 
	/* or call k++ init */
	// kpp(pts, len, cent, n_cluster);

	point tmp = (point) malloc(sizeof(point_t));
	for (clust = 0; clust < n_cluster; clust++) {
		tmp->x = cents.at<float>(clust, 0);
		tmp->y = cents.at<float>(clust, 1);
		tmp->z = cents.at<float>(clust, 2);
		cent[clust] = *tmp;
		// tmp = NULL;
	}

	point tmp_d = (point) malloc(sizeof(point_t));
	for (clust = 0; clust < len; clust++) {
		tmp_d->x = data.at<float>(clust, 0);
		tmp_d->y = data.at<float>(clust, 1);
		tmp_d->z = data.at<float>(clust, 2);
		// std::cout << "Count : " << clust << " : " <<  tmp->x << " : " << tmp->y << " : " <<tmp->z << std::endl;
		data_p[clust] = *tmp_d;
		// tmp = NULL;
	}
	int ctr = 0;
	#	define for_len for (j = 0, p = pts; j < len; j++, p++)
	for_len p->group = nearest(p, cent, n_cluster, 0);
 
	do {
		/* group element for centroids are used as counters */
		for_n { c->group = 0; c->x = c->y = 0; }
		for_len {
			c = cent + p->group;
			c->group++;
			c->x += p->x; c->y += p->y;
		}
		for_n { c->x /= c->group; c->y /= c->group; }
 
		changed = 0;
		/* find closest centroid of each point */
		for_len {
			min_i = nearest(p, cent, n_cluster, 0);
			if (min_i != p->group) {
				changed++;
				p->group = min_i;
			}
		}
	} while (++ctr  < 1); /* stop when 99.9% of points are good */
 
	for_n { c->group = i; }

	for_len{
		std::cout << p->group << std::endl;
	}
 
	return cent;
}

#endif