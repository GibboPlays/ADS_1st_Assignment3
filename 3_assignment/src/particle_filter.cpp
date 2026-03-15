#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include "particle/particle_filter.h"
using namespace std;

static  default_random_engine gen;

/*
* This function initialize randomly the particles
* Input:
*  std - noise that might be added to the position
*  nParticles - number of particles
*/
void ParticleFilter::init_random(double std[],int nParticles) {
    for(int i=0; i<nParticles; i++)
    {
        particles[i].x += std[0];
        particles[i].y += std[1];
        particles[i].theta += std[2];
    }
}

/*
* This function initialize the particles using an initial guess
* Input:
*  x,y,theta - position and orientation
*  std - noise that might be added to the position
*  nParticles - number of particles
*/ 
void ParticleFilter::init(double x, double y, double theta, double std[],int nParticles) {
    num_particles = nParticles;
    normal_distribution<double> dist_x(-std[0], std[0]); //random value between [-noise.x,+noise.x]
    normal_distribution<double> dist_y(-std[1], std[1]);
    normal_distribution<double> dist_theta(-std[2], std[2]);

	for(int i=0; i < nParticles; i++)
    {
        Particle p;
        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);

        particles.push_back(p);
    }
    
    is_initialized=true;
}

/*
* The predict phase uses the state estimate from the previous timestep to produce an estimate of the state at the current timestep
* Input:
*  delta_t  - time elapsed beetween measurements
*  std_pos  - noise that might be added to the position
*  velocity - velocity of the vehicle
*  yaw_rate - current orientation
* Output:
*  Updated x,y,theta position
*/
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    //for each particle
    for (int i=0; i<num_particles; i++)
    {
        double x,y,theta;
        if (fabs(yaw_rate) < 0.00001) {
            x = (velocity * delta_t) * std::cos(particles[i].theta);
            y = (velocity * delta_t) * std::sin(particles[i].theta);
            theta = 0;
        }else{ 
            x = (velocity / yaw_rate)*(std::sin(particles[i].theta + yaw_rate * delta_t) - std::sin(particles[i].theta));
            y = (velocity / yaw_rate)*(std::cos(particles[i].theta) - std::cos(particles[i].theta + yaw_rate * delta_t));
            theta = yaw_rate * delta_t;
        }
        
        normal_distribution<double> dist_x(0, std_pos[0]); //the random noise cannot be negative in this case
        normal_distribution<double> dist_y(0, std_pos[1]);
        normal_distribution<double> dist_theta(0, std_pos[2]);
        
        particles[i].x += x + dist_x(gen);
        particles[i].y += y + dist_y(gen);
        particles[i].theta += theta + dist_theta(gen);

	}
}

/*
* This function associates the landmarks from the MAP to the landmarks from the OBSERVATIONS
* Input:
*  mapLandmark   - landmarks of the map
*  observations  - observations of the car
* Output:
*  Associated observations to mapLandmarks (perform the association using the ids)
*/
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> mapLandmark, std::vector<LandmarkObs>& observations) {
   // Assign to observations[i].id the id of the landmark with the smallest euclidean distance
    for (int i=0; i<observations.size(); i++)
    {
        double minDist = INFINITY;
        int idMin = -1;
        for(int j=0; j<mapLandmark.size(); j++)
        {
            double dist = std::sqrt(std::pow(observations[i].x - mapLandmark[j].x, 2) + std::pow(observations[i].y - mapLandmark[j].y, 2));
            if(dist < minDist) 
            {
                minDist = dist;
                idMin = mapLandmark[j].id;
            }
        }
        observations[i].id = idMin;
    }
}

/*
* This function transform a local (vehicle) observation into a global (map) coordinates
* Input:
*  observation   - A single landmark observation
*  p             - A single particle
* Output:
*  global         - transformation of the observation from local coordinates to global
*/
LandmarkObs transformation(LandmarkObs observation, Particle p){
    LandmarkObs global;
    
    global.id = observation.id;
    global.x = p.x + (std::cos(p.theta) * observation.x) - (std::sin(p.theta) * observation.y);
    global.y = p.y + (std::sin(p.theta) * observation.x) + (std::cos(p.theta) * observation.y);

    return global;
}

/*
* This function updates the weights of each particle
* Input:
*  std_landmark   - Sensor noise
*  observations   - Sensor measurements
*  map_landmarks  - Map with the landmarks
* Output:
*  Updated particle's weight (particles[i].weight *= w)
*/
void ParticleFilter::updateWeights(double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

    //Creates a vector that stores tha map (this part can be improved)
    std::vector<LandmarkObs> mapLandmark;
    for(int j=0;j<map_landmarks.landmark_list.size();j++){
        mapLandmark.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i,map_landmarks.landmark_list[j].x_f,map_landmarks.landmark_list[j].y_f});
    }
    for(int i=0;i<particles.size();i++){

        // Before applying the association we have to transform the observations in the global coordinates
        std::vector<LandmarkObs> transformed_observations;
        //for each observation transform it (transformation function)
        for(int j=0; j<observations.size(); j++)
        {
            transformed_observations.push_back(transformation(observations[j], particles[i]));
        }
        //perform the data association (associate the landmarks to the observations)
        dataAssociation(mapLandmark, transformed_observations);

        particles[i].weight = 1.0;
        // Compute the probability
		//The particles final weight can be represented as the product of each measurement’s Multivariate-Gaussian probability density
		//We compute basically the distance between the observed landmarks and the landmarks in range from the position of the particle
        for(int k=0;k<transformed_observations.size();k++){
            double obs_x,obs_y,l_x,l_y;
            obs_x = transformed_observations[k].x;
            obs_y = transformed_observations[k].y;
            //get the associated landmark 
            for (int p = 0; p < mapLandmark.size(); p++) {
                if (transformed_observations[k].id == mapLandmark[p].id) {
                    l_x = mapLandmark[p].x;
                    l_y = mapLandmark[p].y;
                }
            }	
			// How likely a set of landmarks measurements are, given a prediction state of the car 
            double w = exp( -( pow(l_x-obs_x,2)/(2*pow(std_landmark[0],2)) + pow(l_y-obs_y,2)/(2*pow(std_landmark[1],2)) ) ) / ( 2*M_PI*std_landmark[0]*std_landmark[1] );
            particles[i].weight *= w;
        }

    }    
}

/*
* This function resamples the set of particles by repopulating the particles using the weight as metric
*/
void ParticleFilter::resample() {
    
    uniform_int_distribution<int> dist_distribution(0,num_particles-1);
    double beta  = 0.0;
    vector<double> weights;
    int index = dist_distribution(gen);
    vector<Particle> new_particles;

    for(int i=0;i<num_particles;i++)
        weights.push_back(particles[i].weight);
																
    float max_w = *max_element(weights.begin(), weights.end());
    uniform_real_distribution<double> uni_dist(0.0, max_w);

    for(int i=0;i<num_particles;i++)
    {
        beta += uni_dist(gen);
        while(particles[index].weight < beta) 
        {
            beta -= particles[index].weight;
            index = (index + 1) % num_particles;
        }
        new_particles.push_back(particles[index]);
    }
    particles = new_particles;
}


