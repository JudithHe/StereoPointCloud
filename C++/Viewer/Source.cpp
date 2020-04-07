#include <iostream>
#include <cstdlib>
#include <fstream>
#include <math.h>
#include <SDL.h>

using namespace std;

struct Feature {
	double x, y, z;
	int r, g, b;

	Feature() {}

	Feature(double x, double y, double z, int r, int g, int b) :
		x(x), y(y), z(z), r(r), g(g), b(b) {}
} *features;
int feature_count;

SDL_Window* window;
SDL_Renderer* renderer;

const char* INPUT_ADDRESS = "points.txt";

const int WINDOW_WIDTH = 600;
const int WINDOW_HEIGHT = 600;
const double ROTATION_STEP = 0.1;

void initializeWindow();
void drawRectangle(int x, int y, int w, int h, int r, int g, int b);
void readInput();
void drawFeatures();
void rotateFeaturesX(double sign);
void rotateFeaturesY(double sign);

void initializeWindow()
{
	cout << "Initializing SDL..." << endl;

	SDL_Init(SDL_INIT_EVERYTHING);
	window = SDL_CreateWindow("Stereo Map", 0, 25, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
	renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

	drawRectangle(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, 0, 0);
}

void drawRectangle(int x, int y, int w, int h, int r, int g, int b)
{
	SDL_Rect* rect = new SDL_Rect();
	rect->x = x;
	rect->y = y;
	rect->w = w;
	rect->h = h;

	SDL_SetRenderDrawColor(renderer, r, g, b, 255);
	SDL_RenderFillRect(renderer, rect);
}

void drawPoint(int x, int y, int r, int g, int b)
{
	SDL_SetRenderDrawColor(renderer, r, g, b, 255);
	SDL_RenderDrawPoint(renderer, x, y);
}

void readInput()
{
	cout << "Reading input..." << endl;

	ifstream fin(INPUT_ADDRESS);

	double x, y, z;
	int r, g, b;

	fin >> feature_count;
	features = (Feature*)malloc(feature_count * sizeof(Feature));

	cout << "\t" << feature_count << " input points detected" << endl;

	for (int i = 0; i < feature_count; i++)
	{
		fin >> x >> y >> z >> r >> g >> b;
		
		if (z == 1)
			features[i] = Feature(0, 0, 0, 0, 0, 0);
		else
			features[i] = Feature(x, y, 400 * z, r, g, b);
	}
}

void drawFeatures()
{
	drawRectangle(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, 0, 0);
	for (int i = 0; i < feature_count; i++)
	{
		//drawPoint(features[i].x, features[i].y, features[i].r, features[i].g, features[i].b);
		drawRectangle(features[i].x, features[i].y, 2, 2, features[i].r, features[i].g, features[i].b);
	}
}

void rotateFeaturesX(double sign)
{
	int xc = WINDOW_WIDTH / 2;
	int yc = WINDOW_HEIGHT / 2;

	double zz, yy;

	for (int i = 0; i < feature_count; i++)
	{
		zz = features[i].z;
		yy = features[i].y;

		features[i].y = yc + (yy - yc) * cos(sign * ROTATION_STEP) - zz * sin(sign * ROTATION_STEP);
		features[i].z = (yy - yc) * sin(sign * ROTATION_STEP) + zz * cos(sign * ROTATION_STEP);
	}

	drawFeatures();
	SDL_RenderPresent(renderer);
}

void rotateFeaturesY(double sign)
{
	int xc = WINDOW_WIDTH / 2;
	int yc = WINDOW_HEIGHT / 2;

	double zz, xx;

	for (int i = 0; i < feature_count; i++)
	{
		xx = features[i].x;
		zz = features[i].z;

		features[i].z = zz * cos(sign * ROTATION_STEP) - (xx - xc) * sin(sign * ROTATION_STEP);
		features[i].x = xc + zz * sin(sign * ROTATION_STEP) + (xx - xc) * cos(sign * ROTATION_STEP);
	}

	drawFeatures();
	SDL_RenderPresent(renderer);
}

int main(int argc, char* args[])
{
	initializeWindow();
	readInput();
	drawFeatures();
	
	SDL_RenderPresent(renderer);

	SDL_Event e;
	while (true)
	{
		SDL_PollEvent(&e);

		if (e.type == SDL_QUIT || (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE))
		{
			break;
		}

		if (e.type == SDL_KEYDOWN)
		{
			switch (e.key.keysym.sym)
			{
			case SDLK_LEFT:
				rotateFeaturesY(1);
				break;
			case SDLK_RIGHT:
				rotateFeaturesY(-1);
				break;
			case SDLK_DOWN:
				rotateFeaturesX(1);
				break;
			case SDLK_UP:
				rotateFeaturesX(-1);
				break;
			default:
				break;
			}
		}
	}

	SDL_Quit();
	return 0;
}