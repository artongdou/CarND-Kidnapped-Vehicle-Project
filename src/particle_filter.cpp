/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
#include "multiv_gauss.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::discrete_distribution;
using std::default_random_engine;
using std::cout;
using std::endl;
using std::cerr;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  Particle next_particle;
  default_random_engine eng;
  normal_distribution<double> dist_x{x, std[0]};
  normal_distribution<double> dist_y{y, std[1]};
  normal_distribution<double> dist_theta{theta, std[2]};

  // Set the number of particles
  num_particles = 150;

  for (int idx = 0; idx < num_particles; ++idx) {
    next_particle.id = idx;
    next_particle.x = dist_x(eng);
    next_particle.y = dist_y(eng);
    next_particle.theta = dist_theta(eng);
    next_particle.weight = 1;
    particles.push_back(next_particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  // random generator
  default_random_engine eng;
  // assume normal distribution
  normal_distribution<double> dist_x{0, std_pos[0]};
  normal_distribution<double> dist_y{0, std_pos[1]};
  normal_distribution<double> dist_theta{0, std_pos[2]};

  // predict each particle one by one
  vector<Particle>::iterator it; 
  for (it = particles.begin(); it != particles.end(); ++it) {
    
    double x_hat;
    double y_hat;
    double theta_hat = (*it).theta + yaw_rate*delta_t;

    // avoid divide by zero
    if (yaw_rate < 0.00001) {
      x_hat = (*it).x + velocity*delta_t*cos(theta_hat);
      y_hat = (*it).y + velocity*delta_t*sin(theta_hat);
    } else {
      x_hat = (*it).x + velocity/yaw_rate*(sin(theta_hat)-sin((*it).theta));
      y_hat = (*it).y + velocity/yaw_rate*(cos((*it).theta)-cos(theta_hat));
    }

    // Add Gaussian noise [x,y,theta] and update particle to new location
    (*it).x = x_hat + dist_x(eng);
    (*it).y = y_hat + dist_y(eng);
    (*it).theta = theta_hat + dist_theta(eng);
  }
}

void ParticleFilter::predictLandmarks(const Particle particle, 
                                      const double sensor_range, 
                                      const Map &map_landmarks,
                                      vector<Map::single_landmark_s>& predicted_landmarks) {
  
  Map::single_landmark_s landmark;
  bool empty = true;

  for (int i = 0; i < map_landmarks.landmark_list.size(); ++i) {

    landmark = map_landmarks.landmark_list[i];

    if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) <= sensor_range) {

      // save the closeby landmark within sensor range
      predicted_landmarks.push_back(landmark);

      // At least one landmark closeby detected
      empty = false;
    }
  }

  // Detect empty predicted landmarks for debugging
  if (empty) {
    cerr << "empty predicted landmarks!!!!!!!" << endl;
    cerr << particle.x << endl << particle.y << endl;
  }
}

void ParticleFilter::transformObservation(const Particle particle, 
                                          const vector<LandmarkObs> &observations,
                                          vector<LandmarkObs>& transformed_obs) {
  LandmarkObs t_obs, obs;

  // Read [x, y, theta] of the particle
  double x_part = particle.x;
  double y_part = particle.y;
  double theta = particle.theta;

  for (int k = 0; k < observations.size(); ++k) {
    obs = observations[k];
    t_obs.x = obs.x*cos(theta) - obs.y*sin(theta) + x_part;
    t_obs.y = obs.x*sin(theta) + obs.y*cos(theta) + y_part;
    transformed_obs.push_back(t_obs);
  }
}

void ParticleFilter::dataAssociation(vector<Map::single_landmark_s> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  LandmarkObs* o;
  Map::single_landmark_s p;
  double dist_to_landmark;

  for (int i = 0; i < observations.size(); ++i) {
    // current observation in the list
    o = &(observations[i]);

    // initialize minimum distance
    double min_dist = std::numeric_limits<double>::max();

    // find the closest landmark in the list of predicted landmarks
    for (int j = 0; j < predicted.size(); ++j) {
      p = predicted[j];
      // calculate distance to landmark
      dist_to_landmark = dist(o->x, o->y, p.x_f, p.y_f);

      if (dist_to_landmark < min_dist) {
        // assign landmark id to the observation
        o->id = p.id_i;
        min_dist = dist_to_landmark;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  vector<Map::single_landmark_s> predicted_landmarks;
  vector<LandmarkObs> transformed_obs;
  vector<int> associations;
  vector<double> sense_x;
  vector<double> sense_y;
  
  for (int idx = 0; idx < particles.size(); ++idx) {

    // predict landmarks within sensor range
    predicted_landmarks.clear();
    predictLandmarks(particles[idx], sensor_range, map_landmarks, predicted_landmarks);

    // transform observations to world coordinates
    transformed_obs.clear();
    transformObservation(particles[idx], observations, transformed_obs);

    // associate landmarks
    dataAssociation(predicted_landmarks, transformed_obs);
    SetAssociations(particles[idx], transformed_obs);
    
    // calculate new particle weights
    double weight = 1.0;
    for (int n = 0; n < particles[idx].associations.size(); ++n) {
      const Map::single_landmark_s landmark = map_landmarks.landmark_list[particles[idx].associations[n] - 1];
      weight *= multiv_prob(
                std_landmark[0],
                std_landmark[1],
                particles[idx].sense_x[n],
                particles[idx].sense_y[n],
                landmark.x_f,
                landmark.y_f);
    }

    // udpate particle weight
    particles[idx].weight = weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  int num_particles = particles.size();
  default_random_engine eng;
  vector<double> weights;
  vector<Particle> resampled;

  // reconstruct a vector of weights
  weights.reserve(num_particles);
  for (int i = 0; i < num_particles; ++i) {
    weights.push_back(particles[i].weight);
  }

  // discrete distribution based on weights
  discrete_distribution d(weights.begin(), weights.end());

  // pick N particles randomly with replacement
  for (int i = 0; i < num_particles; ++i) {
    int random_idx = d(eng);
    resampled.push_back(particles[random_idx]);
  }

  // update particles to be resampled particles
  particles = resampled;
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<LandmarkObs>& transformed_obs) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  vector<int> associations;
  vector<double> sense_x;
  vector<double> sense_y;

  for (int i = 0; i < transformed_obs.size(); ++i) {
    associations.push_back(transformed_obs[i].id);
    sense_x.push_back(transformed_obs[i].x);
    sense_y.push_back(transformed_obs[i].y);
  }

  // update the transformed observations in particle
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}