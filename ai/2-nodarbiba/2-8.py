from __future__ import annotations

import time
from telnetlib import GA
import numpy as np
import os
import matplotlib
import matplotlib.backend_bases

if os.name == "darwin":
    matplotlib.use("MacOSX")  # for mac
else:
    matplotlib.use("TkAgg")  # for unix/windows

import matplotlib.pyplot as plt

SPACE_SIZE = (9, 9)
plt.rcParams["figure.figsize"] = (15, 15)
plt.ion()  # interactive mode
plt.style.use("dark_background")


def rotation_mat(degrees):
    theta = np.radians(degrees)
    cos_val = np.cos(theta)
    sin_val = np.sin(theta)
    return np.array([
        [cos_val, -sin_val, 0],
        [sin_val, cos_val, 0],
        [0, 0, 1]
    ])


def translation_mat(dx, dy):
    return np.array([
        [1, 0, dx],
        [0, 1, dy],
        [0, 0, 1]
    ])


def scale_mat(sx, sy):
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])


def scew_mat(x, y):
    return np.array([
        [1, y, 0],
        [x, 1, 0],
        [0, 0, 1]
    ])


def dot(a, b):
    is_transposed = False

    a = np.atleast_2d(a)  # [1,2,3] -> [[1,2,3]]
    b = np.atleast_2d(b)  # [1,2,3] -> [[1,2,3]]

    if a.shape[1] != b.shape[0]:
        is_transposed = True
        b = np.transpose(b)

    a_rows = a.shape[0]
    b_columns = b.shape[1]

    product = np.zeros((a_rows, b_columns))

    for i in range(a_rows):
        for j in range(b_columns):
            product[i, j] += np.sum(a[i, :] * b[:, j])

    if is_transposed:
        product = np.transpose(product)

    if product.shape[0] == 1:
        product = product.flatten()

    return product


def vec2d_to_vec3d(vec2):
    I = np.array([
        [1, 0],
        [0, 1],
        [0, 0]
    ])
    vec3 = np.dot(I, vec2) + np.array([0, 0, 1])
    return vec3


def vec3d_to_vec2d(vec3):
    I = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])
    vec2 = np.dot(I, vec3)
    return vec2


class StaticObject:
    def __init__(self, vec_pos, t_centered=np.identity(3)):
        self.vec_pos: np.ndarray = vec_pos.astype(np.float)
        self.vec_dir_init = np.array([0.0, 1.0])
        self.vec_dir = np.array(self.vec_dir_init)
        self.geometry: list[np.ndarray] = []
        self.__angle: float = 0
        self.color = 'r'

        self.C = np.identity(3)
        self.R = np.identity(3)
        self.S = np.identity(3)
        self.T = np.identity(3)
        self.T_centered = t_centered

        self.update_transformation()

    def set_angle(self, angle):
        self.__angle = angle
        self.R = rotation_mat(angle)

        vec3d = vec2d_to_vec3d(self.vec_dir_init)
        vec3d = dot(self.R, vec3d)
        self.vec_dir = vec3d_to_vec2d(vec3d)

        self.update_transformation()

    def get_angle(self) -> float:
        return self.__angle

    def update_movement(self, delta_time: float):
        pass

    def update_transformation(self):
        self.T = translation_mat(self.vec_pos[0], self.vec_pos[1])

        self.C = np.identity(3)
        self.C = self.C @ self.T
        self.C = self.C @ self.R
        self.C = self.C @ self.T_centered
        self.C = self.C @ self.S

    def get_polygon_vertices(self):
        x_values = []
        y_values = []

        for vec2d in self.geometry:
            vec3d = vec2d_to_vec3d(vec2d)
            vec3d = dot(self.C, vec3d)
            vec2d = vec3d_to_vec2d(vec3d)

            x_values.append(vec2d[0])
            y_values.append(vec2d[1])

        return x_values, y_values

    def draw(self):
        x_values, y_values = self.get_polygon_vertices()
        plt.plot(x_values, y_values, c=self.color)


class Planet(StaticObject):

    def __init__(self, vec_pos: np.ndarray = None):
        super().__init__(vec_pos)

        self.color = 'b'
        self.vec_F = np.zeros(2, )  # panet gravity force

        step = 2 * np.pi / 20
        self.radius = np.random.random() * 0.5 + 1.0
        self.geometry = []
        theta = 0
        while theta < 2 * np.pi:
            self.geometry.append(np.array([
                np.cos(theta) * self.radius,
                np.sin(theta) * self.radius
            ]))
            theta += step
        self.geometry.append(np.array(self.geometry[0]))
        self.update_transformation()

    def update_movement(self, delta_time: float):
        player_pos = Player.get_instance().vec_pos
        dist_vec = self.vec_pos - player_pos
        vec_dir = dist_vec / np.linalg.norm(dist_vec)
        dist = np.sqrt(dist_vec[0] ** 2 + dist_vec[1] ** 2)
        gravity = (100 * self.radius) / (dist ** 2)
        self.vec_F = np.array([gravity * vec_dir[0], gravity * vec_dir[1]])

    # override draw method, implement Mid-Point Circle Drawing Algorithm
    def draw(self):
        x_values, y_values = self.get_polygon_vertices()
        plt.plot(x_values, y_values, c=self.color)


class MovableObject(StaticObject):
    def __init__(self, vec_pos, t_centered=np.identity(3)):
        super().__init__(vec_pos, t_centered)
        self.speed = 0

    def update_movement(self, delta_time: float):
        self.vec_pos += self.vec_dir * self.speed * delta_time
        self.update_transformation()

        if abs(self.vec_pos[0]) > SPACE_SIZE[0]:
            self.vec_pos[0] = -self.vec_pos[0] / abs(self.vec_pos[0]) * SPACE_SIZE[0]
        if abs(self.vec_pos[1]) > SPACE_SIZE[1]:
            self.vec_pos[1] = -self.vec_pos[1] / abs(self.vec_pos[1]) * SPACE_SIZE[1]


class Asteroid(MovableObject):
    def __init__(self, vec_pos):
        super().__init__(vec_pos)

        self.color = 'g'

        step = 2 * np.pi / 20
        self.radius = np.random.random() * 0.2 + 0.2
        self.geometry = []
        theta = 0
        while theta < 2 * np.pi:
            radius = self.radius + np.random.random() * 0.1 - 0.05
            self.geometry.append(np.array([
                np.cos(theta) * radius,
                np.sin(theta) * radius
            ]))
            theta += step
        self.geometry.append(np.array(self.geometry[0]))

        self.S = scew_mat(x=0.2 * np.random.random(), y=0.2 * np.random.random())

        self.speed = np.random.random() * 20 + 10
        self.set_angle(np.random.random() * 360)
        self.update_transformation()

    def update_movement(self, delta_time: float):
        super().update_movement(delta_time)


class Rocket(MovableObject):
    def __init__(self, vec_pos):
        super().__init__(vec_pos)

        self.color = 'y'

        self.geometry = np.array([
            [0, -0.1],
            [0, 0.1],
        ])

        self.pos = np.array(Player.get_instance().vec_pos)
        self.speed = 60 + Player.get_instance().speed
        self.set_angle(Player.get_instance().get_angle())
        self.update_transformation()

    def update_movement(self, delta_time: float):
        super().update_movement(delta_time)


class Player(MovableObject):
    _instance: Player = None

    def __init__(self, vec_pos):
        t_centered = translation_mat(dx=0, dy=-0.5)

        super().__init__(vec_pos, t_centered)
        self.geometry = np.array([
            [-1, 0],
            [1, 0],
            [0, 1],
            [-1, 0]
        ])

        self.S = scale_mat(sx=0.5, sy=1)
        self.speed = 0
        self.update_transformation()

        if not Player._instance:
            Player._instance = self
        else:
            raise Exception("Cannot construct singleton twice")

    def activate_thrusters(self):
        self.speed += 50.0

    def fire_rocket(self):
        rocket = Rocket(self.vec_pos)
        Game.get_instance().actors.append(rocket)

    def update_movement(self, delta_time: float):
        self.speed -= delta_time * 30.0
        self.speed = max(0, self.speed)

        for actor in Game.get_instance().actors:
            if isinstance(actor, Planet):
                self.vec_pos += actor.vec_F * delta_time

        super().update_movement(delta_time)

    @staticmethod
    def get_instance() -> Player:
        if not Player._instance:
            Player()
        return Player._instance


def three_point_orientation(p1, p2, p3):
    val = (p2[1] - p1[1]) * (p3[0] - p2[0]) - (p2[0] - p1[0]) * (p3[1] - p2[1])
    if val == 0:
        return 0  # collinear
    elif val > 0:
        return 1  # clockwise
    else:
        return 2  # counterclockwise


def on_segment(p, q, r):
    if max(p[0], r[0]) >= q[0] >= min(p[0], r[0]) and \
            max(p[1], r[1]) >= q[1] >= min(p[1], r[1]):
        return True
    return False


def are_segments_colliding(a, b) -> bool:
    o1 = three_point_orientation(a[0], a[1], b[0])
    o2 = three_point_orientation(a[0], a[1], b[1])
    o3 = three_point_orientation(b[0], b[1], a[0])
    o4 = three_point_orientation(b[0], b[1], a[1])

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(a[0], b[0], a[1]):
        return True
    if o2 == 0 and on_segment(a[0], b[1], a[1]):
        return True
    if o3 == 0 and on_segment(b[0], a[0], b[1]):
        return True
    if o4 == 0 and on_segment(b[0], a[1], b[1]):
        return True

    return False


def are_polygons_colliding(a, b) -> bool:
    a_segments = [(a[i], a[i + 1]) for i in range(len(a) - 1)]
    b_segments = [(b[i], b[i + 1]) for i in range(len(b) - 1)]
    for a_segment in a_segments:
        for b_segment in b_segments:
            if are_segments_colliding(a_segment, b_segment):
                return True
    return False


def get_poly_center(polygon):
    return np.mean(polygon, axis=0)


class Game:
    _instance: Game = None

    def __init__(self):
        super(Game, self).__init__()
        self.is_running = True
        self.score = 0
        self.lives = 0

        self.actors: list[StaticObject] = [
            Player(vec_pos=np.array([0, 0])),
            Planet(vec_pos=np.array([-7, -3])),
            Planet(vec_pos=np.array([8, -4])),
        ]

        for _ in range(5):
            asteroid = Asteroid(vec_pos=np.array(
                [
                    np.random.randint(-SPACE_SIZE[0], SPACE_SIZE[0]),
                    np.random.randint(-SPACE_SIZE[1], SPACE_SIZE[1]),
                ]
            ))
            self.actors.append(asteroid)

        if not Game._instance:
            Game._instance = self
        else:
            raise Exception("Cannot construct singleton twice")

    def press(self: Game, event: matplotlib.backend_bases.Event):
        player = Player.get_instance()
        print("press", event.key)
        if event.key == "escape":
            self.is_running = False
        elif event.key == "right":
            player.set_angle(player.get_angle() - 5)
        elif event.key == "left":
            player.set_angle(player.get_angle() + 5)
        elif event.key == "up":
            player.activate_thrusters()
        elif event.key == " ":
            player.fire_rocket()

    def on_close(self: Game, event: matplotlib.backend_bases.Event):
        self.is_running = False

    def main(self: Game):

        fig, _ = plt.subplots()
        fig.canvas.mpl_connect("key_press_event", self.press)
        fig.canvas.mpl_connect("close_event", self.on_close)
        dt = 1e-3

        while self.is_running:
            plt.clf()
            plt.axis("off")

            plt.title(
                f"score: {self.score} speed: {round(Player.get_instance().speed, 1)} pos:  {Player.get_instance().vec_pos}")
            plt.tight_layout(pad=0)

            plt.xlim(-10, 10)
            plt.ylim(-10, 10)

            for actor in self.actors:  # polymorphism
                actor.update_movement(dt)
                actor.draw()

            # actor polygons
            actor_polygons = [list(zip(*actor.get_polygon_vertices())) for actor in self.actors]
            actor_poly_centers = [get_poly_center(list(zip(*actor.get_polygon_vertices()))) for actor in self.actors]

            # collision detection
            destroyed_actors = set()
            for i in range(len(self.actors)):
                i_poly = actor_polygons[i]
                i_center = actor_poly_centers[i]
                for j in range(i + 1, len(self.actors)):
                    j_center = actor_poly_centers[j]
                    i_j_dist = np.linalg.norm(i_center - j_center)
                    if i_j_dist > 5:
                        continue
                    j_poly = actor_polygons[j]
                    if are_polygons_colliding(i_poly, j_poly):
                        # player can't run into planet or asteroid
                        i_planet_asteroid = isinstance(self.actors[i], Planet) or isinstance(self.actors[i], Asteroid)
                        j_planet_asteroid = isinstance(self.actors[j], Planet) or isinstance(self.actors[j], Asteroid)
                        if isinstance(self.actors[i], Player) and j_planet_asteroid or \
                                isinstance(self.actors[j], Player) and i_planet_asteroid:
                            self.is_running = False
                            plt.draw()
                            print("Game over")
                            time.sleep(3)
                            exit(0)

                        # rockets can destroy asteroids
                        if isinstance(self.actors[i], Rocket) and isinstance(self.actors[j], Asteroid):
                            destroyed_actors.add(self.actors[j])
                            self.score += 1
                        elif isinstance(self.actors[j], Rocket) and isinstance(self.actors[i], Asteroid):
                            destroyed_actors.add(self.actors[i])
                            self.score += 1

            # remove destroyed actors
            for actor in destroyed_actors:
                self.actors.remove(actor)

            plt.draw()
            plt.pause(dt)

    @staticmethod
    def get_instance() -> Game:
        if not Game._instance:
            Game()
        return Game._instance


game = Game.get_instance()
game.main()
