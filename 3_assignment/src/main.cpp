#include <chrono>
#include <iostream>
#include <math.h>
#include <fstream>
#include "particle/particle_filter.h"
#include <pcl/common/transforms.h>
#include "Renderer.hpp"
#include <pcl/filters/voxel_grid.h>
#include <particle/helper_cloud.h>
#include <pcl/io/pcd_io.h>
#include <thread>

#define NPARTICLES 200
#define circleID "circle_id"
#define reflectorID "reflector_id"

using namespace std;
using namespace lidar_obstacle_detection;


Map map_mille;  
ParticleFilter pf;
Renderer renderer;
std::ofstream myfile;
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_particles(new pcl::PointCloud<pcl::PointXYZ>);
vector<Particle> best_particles;

// Parametri del Filtro
double sigma_init [3] = {0.1, 0.1, 0.1}; 
double sigma_pos [3]  = {0.05, 0.05, 0.05}; 
double sigma_landmark [2] = {0.4, 0.4};    
std::vector<Color> colors = {Color(1,0,0), Color(1,1,0), Color(0,0,1), Color(1,0,1), Color(0,1,1)};

// Aggiorna la posizione delle particelle nel visualizzatore
void showPCstatus(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, vector<Particle> particles){
    for (size_t i = 0; i < particles.size(); ++i) {
        cloud->points[i].x = particles[i].x;
        cloud->points[i].y = particles[i].y;
    }
    renderer.updatePointCloud(cloud, "particles");
}

int main(int argc, char **argv)
{
    //inizializzazione visualizzatore
    renderer.InitCamera(CameraAngle::XY);
    renderer.ClearViewer();

    //caricamento mappa
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudMap (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudReflectors (new pcl::PointCloud<pcl::PointXYZ>);
    
    if (pcl::io::loadPCDFile ("../data/map_reflector.pcd", *cloudReflectors) == -1 ||
        pcl::io::loadPCDFile ("../data/map_pepperl.pcd", *cloudMap) == -1) {
        cout << "Errore caricamento file .pcd in ../data/" << endl;
        return -1;
    }

    read_map_data("../data/map_data.txt", map_mille);

    renderer.RenderPointCloud(cloudMap, "originalCloud", colors[2]);
    renderer.RenderPointCloud(cloudReflectors, "reflectorCloud", colors[0]);

    //inizializzazione filtro
    double GPS_x = 2.0, GPS_y = 1.0, GPS_theta = -1.0;
    pf.init(GPS_x, GPS_y, GPS_theta, sigma_init, NPARTICLES);

    for(int i=0; i<NPARTICLES; i++){
        cloud_particles->push_back(pcl::PointXYZ(pf.particles[i].x, pf.particles[i].y, 0));
    }   
    renderer.RenderPointCloud(cloud_particles, "particles", colors[1]);

    myfile.open("./res.txt");

    //ciclo simulazione
    int step = 0;
    while (!renderer.WasViewerStopped() && step < 1000) {
        
        auto t_start = std::chrono::high_resolution_clock::now();

        //prediction (Simuliamo velocità 0.5 m/s e rotazione 0.02 rad/s)
        pf.prediction(0.1, sigma_pos, 0.5, 0.02);

        //update
        vector<LandmarkObs> observations; 
        //qui si userrebbe extractReflectors(cloud) in caso di veri sensori
        
        pf.updateWeights(sigma_landmark, observations, map_mille);

        //resemple
        pf.resample();

        //trova particella migliore per la visualizzazione
        Particle best_p = pf.particles[0];
        for(auto& p : pf.particles) {
            if(p.weight > best_p.weight) best_p = p;
        }

        //aggiornamento grafica
        showPCstatus(cloud_particles, pf.particles);
        
        //disegna cerchio cerchio rosso su posizione stimata
        renderer.removeShape("best_pos");
        renderer.addCircle(best_p.x, best_p.y, "best_pos", 0.3, 1, 0, 0);
        
        renderer.SpinViewerOnce();
        //pausa
        //std::this_thread::sleep_for(std::chrono::milliseconds(50));

        auto t_end = std::chrono::high_resolution_clock::now();
        double delta_t = std::chrono::duration<double>(t_end - t_start).count();
        
        myfile << best_p.x << " " << best_p.y << " " << delta_t << endl;

        step++;
    }

    cout << "Simulazione terminata. Chiudi la finestra del viewer per uscire dal programma." << endl;
    
    //chiude programma quando chiudi finestra
    while (!renderer.WasViewerStopped()) {
        renderer.SpinViewerOnce();
    }

    myfile.close();
    return 0;
}