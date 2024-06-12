import yaml
import time
import networkx as nx
import osmnx as ox


from nuplane.core import XPlaneCore
from nuplane.core import kill_all_servers
from nuplane.utils.transform import get_bearing


from nuplane.env import NUPlaneEnv
from nuplane.utils.transform import ft_to_m
from nuplane.utils.airport_parser import (
    parse_airport_route_network,
    parse_airport_network,
)
from nuplane.maps import Map

from experiments.auto_landing.auto_landing_experiment import AutoLandingExperiment
from experiments.auto_landing.glideslope_controller import GlideSlopeController
from experiments.taxiing.control.controllers import PathController
from experiments.taxiing.navigation.path_planner import PathPlanner
from experiments.taxiing.taxi_experiment import TaxiingExperiment
from experiments.agents import Hero, AIAircraft


from data_collector.data_collector import DataCollector
from data_collector.experiments.simple_experiment import SimpleExperiment


import matplotlib.pyplot as plt

from utils import skip_run

# Run the simulation
config = yaml.load(open("config/xplane_config.yaml"), Loader=yaml.SafeLoader)

with skip_run("skip", "collect_data") as check, check():
    kill_all_servers()
    core = XPlaneCore(config)

    core.client.sendDREF("sim/weather/cloud_type[0]", 1)
    time.sleep(5)
    core.client.sendDREF("sim/weather/cloud_type[0]", 0)
    time.sleep(5)
    core.client.sendDREF("sim/weather/cloud_type[0]", 1)

    core.client.sendDREF("sim/time/zulu_time_sec", 43200)

    time.sleep(10)
    kill_all_servers()

with skip_run("skip", "explore_positioning") as check, check():
    # kill_all_servers()
    core = XPlaneCore(config)
    # core.client.pauseSim(False)
    start = 0
    end = 1062
    experiment_config = {"gate": "B6"}
    position = core.map.taxi_network.gates[experiment_config["gate"]]
    position_1 = core.map.node_graph.nodes[1038]
    position_2 = core.map.node_graph.nodes[1037]
    position_3 = core.map.node_graph.nodes[1058]

    bearing = get_bearing(
        position_1["lat"], position_1["lon"], position_2["lat"], position_2["lon"]
    )
    core.hero.set_position(position_2["lat"], position_2["lon"], heading=bearing, ac=0)
    core.hero.set_position(position_1["lat"], position_1["lon"], heading=bearing, ac=1)
    core.hero.set_position(position_3["lat"], position_3["lon"], heading=bearing, ac=3)
    pos = position_2["lon"]
    for i in range(100):
        pos = pos + 0.00001
        core.hero.set_position(position_2["lat"], pos, heading=bearing, ac=0)

with skip_run("skip", "explore_route_network") as check, check():
    core = XPlaneCore(config, debug=True)
    G = core.map.get_node_graph()

    spawn_points = yaml.load(open("config/spawn_points.yaml"), Loader=yaml.SafeLoader)
    num = 6
    start = spawn_points[num]["start"]
    end = spawn_points[num]["end"]

    start_node = ox.distance.nearest_nodes(G, X=start["x"], Y=start["y"])
    end_node = ox.distance.nearest_nodes(G, X=end["x"], Y=end["y"])

    route = nx.shortest_path(G, start_node, end_node, weight="length")
    path_planner = PathPlanner(core.map)
    new_lat_lon = path_planner.find_path(start_node, end_node, n_splits=1)

    fig, ax = ox.plot_graph_route(G, route, show=False, close=False)
    ax.plot(new_lat_lon[:, 1], new_lat_lon[:, 0], marker="o")
    plt.show()

with skip_run("skip", "explore_route_network") as check, check():
    airport_id = config["experiment"]["experiment_config"]["airport"]
    G = parse_airport_route_network(airport_id)

    airport_map = Map(client=None, config=config)

    airport_map._setup_taxi_network(config)
    test_graph = airport_map._convert_taxi_network_to_osmnx_graph(
        airport_map.taxi_network, airport_id
    )
    test_G = airport_map._convert_taxi_network_to_graph(airport_map.taxi_network)
    fig, ax = ox.plot_graph(test_G, show=False, close=False)
    for u, v, data in test_G.edges(data=True):
        edge_label = data["name"]
        ax.annotate(
            edge_label,
            xy=(data["start_node_coord"]["lon"], data["start_node_coord"]["lat"]),
            xytext=(3, 3),
            textcoords="offset points",
            color="r",
        )

    spawn_points = yaml.load(open("config/spawn_points.yaml"), Loader=yaml.SafeLoader)
    num = 8
    start = spawn_points[num]["start"]
    end = spawn_points[num]["end"]

    start_node = ox.distance.nearest_nodes(G, X=start["x"], Y=start["y"])
    end_node = ox.distance.nearest_nodes(G, X=end["x"], Y=end["y"])

    # route = nx.shortest_path(G, start_node, end_node, weight="length")
    # path_planner = PathPlanner(airport_map)
    # new_lat_lon = path_planner.find_path(start_node, end_node, n_splits=1)

    # fig, ax = ox.plot_graph_route(G, route, show=False, close=False)
    # ax.plot(new_lat_lon[:, 1], new_lat_lon[:, 0], marker="o", zorder=2)
    # # plt.show()

    G = parse_airport_network(airport_id, feature="runways")
    G.plot(color="y", ax=ax, zorder=1)

    # Plot taxiway and runways
    G = parse_airport_network(airport_id, feature="pavements")
    G.plot(color="g", ax=ax, zorder=0, categorical=True, cmap="RdPu")

    plt.show()

    # fig, ax = ox.plot_graph_route(G, route, show=False, close=False)
    # ax.plot(new_lat_lon[:, 1], new_lat_lon[:, 0], marker="o")
    # plt.show()

with skip_run("skip", "explore_new_route_network") as check, check():
    core = XPlaneCore(config, debug=True)
    G = core.map.get_node_graph()

    spawn_points = yaml.load(open("config/spawn_points.yaml"), Loader=yaml.SafeLoader)
    num = 6
    start = spawn_points[num]["start"]
    end = spawn_points[num]["end"]

    start_node = ox.distance.nearest_nodes(G, X=start["x"], Y=start["y"])
    print(start_node)
    end_node = ox.distance.nearest_nodes(G, X=end["x"], Y=end["y"])
    route = nx.shortest_path(G, start_node, end_node, weight="length")

    path_planner = PathPlanner(core.map)
    new_lat_lon = path_planner.find_path(start_node, end_node, n_splits=1)

    path_planner.set_route(new_lat_lon)
    print(path_planner.distance_tol)

    fig, ax = ox.plot_graph_route(G, route, show=False, close=False)
    ax.plot(new_lat_lon[:, 1], new_lat_lon[:, 0], marker="o")
    plt.show()

with skip_run("skip", "explore_path_planning") as check, check():
    config["experiment"]["type"] = TaxiingExperiment
    config["experiment"]["experiment_config"] = yaml.load(
        open("experiments/taxiing/experiment_config.yaml"), Loader=yaml.SafeLoader
    )

    core = XPlaneCore(config, debug=True)
    G = core.map.get_node_graph()

    spawn_points = yaml.load(open("config/spawn_points.yaml"), Loader=yaml.SafeLoader)
    num = 6
    start = spawn_points[num]["start"]
    end = spawn_points[num]["end"]

    start_node = ox.distance.nearest_nodes(G, X=start["x"], Y=start["y"])
    end_node = ox.distance.nearest_nodes(G, X=end["x"], Y=end["y"])
    config["hero_config"]["spawn_location"] = [start_node]

    # Path planning
    path_planner = PathPlanner(core.map)
    route_lat_lon = path_planner.find_path(start_node, end_node, n_splits=1)
    path_planner.set_route(route_lat_lon)

    hero = Hero(core.client, config["hero_config"])
    core.setup_experiment(
        experiment_config=config["experiment"]["experiment_config"], hero=hero
    )
    core.reset(path_planner.get_orientation())

    # Controller
    controller = PathController(agent=hero)
    hero.apply_action([0, 0, 0, 0, 0, 0, 1.5, 1])

    while not path_planner.get_done_status():
        obs = hero.get_observation()

        # Get heading and distance to next waypoint
        if obs is not None:
            heading = path_planner.get_heading(obs)
            distance_to_next_pos = path_planner.get_distance_to_next_pos()
            control = controller.get_control(distance_to_next_pos, heading)

            # Apply control
            hero.apply_action(control)

    hero.apply_action([0, 0, 0, 0, 0, 0, 1.5, 1])

with skip_run("skip", "taxiing_experiment") as check, check():
    # Setup the environment and experiment
    config["experiment"]["type"] = TaxiingExperiment
    config["experiment"]["experiment_config"] = yaml.load(
        open("experiments/taxiing/experiment_config.yaml"), Loader=yaml.SafeLoader
    )
    xplane_env = NUPlaneEnv(config)
    obs, reward, done, info = xplane_env.reset()

    # Controller or RL
    controller = PathController(agent=xplane_env.experiment.hero)
    i = 0

    t = 1000
    xplane_env.core.client.sendDREF("sim/weather/rain_percent", 1.0)
    xplane_env.core.client.sendDREF("sim/weather/cloud_base_msl_m[1]", 750000)
    xplane_env.core.client.sendDREF("sim/graphics/scenery/airport_light_level", 1)

    while not done:
        i += 1
        control = controller.get_control(obs[0], obs[1])

        # Apply control
        obs, reward, done, info = xplane_env.step(control)

        if obs[2] == 50:
            path = "Custom Scenery/Simple_Ground_Equipment_and_Services_v67.8/Simple_Ground_Equipment_and_Services/MisterX_Lib/Stairs/Generic.obj"
            xplane_env.core.client.loadOBJ(path=path, on_ground=1)

with skip_run("skip", "image_data_feed") as check, check():
    config["experiment"]["type"] = TaxiingExperiment
    config["experiment"]["experiment_config"] = yaml.load(
        open("experiments/taxiing/experiment_config.yaml"), Loader=yaml.SafeLoader
    )

    import matplotlib.pyplot as plt

    core = XPlaneCore(config, debug=True)

    while 1:
        image = core.client.getIMG()
        plt.imshow(image, origin="lower")
        plt.pause(0.0001)

with skip_run("skip", "taxiing_experiment_video") as check, check():
    # Setup the environment and experiment
    config["experiment"]["type"] = TaxiingExperiment
    config["experiment"]["experiment_config"] = yaml.load(
        open("experiments/taxiing/experiment_config.yaml"), Loader=yaml.SafeLoader
    )
    xplane_env = NUPlaneEnv(config, debug=True)
    obs, reward, done, info = xplane_env.reset()

    # Controller or RL
    controller = PathController(agent=xplane_env.experiment.hero)
    i = 0

    t = 100
    xplane_env.core.client.sendDREF("sim/weather/rain_percent", 1.0)
    xplane_env.core.client.sendDREF("sim/weather/cloud_base_msl_m[1]", 750000)
    xplane_env.core.client.sendDREF("sim/graphics/scenery/airport_light_level", 1)

    while not done:
        i += 1
        control = controller.get_control(obs[0], obs[1])

        print(i)

        # Apply control
        obs, reward, done, info = xplane_env.step(control)

        # if i > t and i < (2 * t):
        #     xplane_env.core.client.sendDREF("sim/weather/cloud_type[0]", 4)

        # if i > (2 * t) and i < (3 * t):
        #     xplane_env.core.client.sendDREF("sim/weather/cloud_type[1]", 1)

        # if i > (3 * t) and i < (4 * t):
        #     xplane_env.core.client.sendDREF("sim/weather/cloud_type[0]", 3)

        # if i > (4 * t) and i < (5 * t):
        #     xplane_env.core.client.sendDREF("sim/weather/cloud_type[0]", 2)

with skip_run("skip", "image_data_feed") as check, check():
    config["experiment"]["type"] = TaxiingExperiment
    config["experiment"]["experiment_config"] = yaml.load(
        open("experiments/taxiing/experiment_config.yaml"), Loader=yaml.SafeLoader
    )

    import matplotlib.pyplot as plt

    core = XPlaneCore(config, debug=True)

    while 1:
        image = core.client.getIMG()
        plt.imshow(image, origin="lower")
        plt.pause(0.0001)

with skip_run("skip", "explore_functionality") as check, check():
    # kill_all_servers()
    core = XPlaneCore(config)
    core.map.draw_map(with_labels=True)

    core.client.pauseSim(True)

    core.client.sendDREFs(["sim/operation/fix_all_systems"], [1])

    hero = Hero(core.client)
    ai_carft = AIAircraft(core.client, id=1)

    experiment_config = config["experiment_config"]
    # Position
    position = core.map.gates[experiment_config["gate"]]

    position = core.map.node_graph.nodes[1063]
    position["heading"] = 270.0
    hero.reset()
    hero.set_position(position)

    position = core.map.node_graph.nodes[1062]
    position["heading"] = 270.0
    ai_carft.reset()
    ai_carft.set_position(position, hero.get_altitude())
    core.client.pauseSim(False)

    for i in range(5):
        ai_carft.apply_action([0, 0, 0, 0.5, False])
        hero.apply_action([0, 0, 0, 0.5, False])
        time.sleep(10)
        hero.apply_action([0, 0, 0, 0, False])
        ai_carft.apply_action([0, 0, 0, 0, False])

with skip_run("skip", "data_collector") as check, check():
    config = yaml.load(open("data_collector/sinusoidal.yaml"), Loader=yaml.SafeLoader)

    config["experiment"]["type"] = SimpleExperiment

    data_collector = DataCollector(config=config, write_path=None)
    data_collector.write_loop()

with skip_run("skip", "auto_land_kmwh") as check, check():
    # First launch X-Plane and start a Cessna 172SP at Grant County International Airport (KMWH) Runway 04
    # Note: you can also choose another airport and update the experiment configuration accordingly
    experiment_config = yaml.load(
        open("experiments/auto_landing/experiment_config.yaml"), Loader=yaml.SafeLoader
    )
    config["experiment"] = {
        "type": AutoLandingExperiment,
        "experiment_config": experiment_config,
    }
    print(
        f"Airport set to {config['experiment']['experiment_config']['airport']} by configuration."
    )
    core = XPlaneCore(config)

    try:
        exp = AutoLandingExperiment(config, core)
        exp.reset()

        dt = experiment_config["sim_config"]["dt"]
        gsc = GlideSlopeController(dt)
        gamma = 3.0
        tch = ft_to_m(experiment_config["hero_config"]["tch"])
        runway_elev = exp.actor.runway_elev
        des_u = 50.0
        gsc.set_reference([gamma, tch, runway_elev, des_u])

        # land the plane using full state knowledge
        max_steps = 10000
        for step in range(max_steps):
            obs, _ = exp.get_observation({}, core)
            action = gsc.get_input(obs)
            exp.apply_actions(action, core)
            if exp.get_done_status(obs, core):
                print("Successfully landed.")
                time.sleep(5)
                core.client.pauseSim(True)
                break
        if step >= max_steps:
            print("Ran out of time.")
    except KeyboardInterrupt:
        print("Stopping due to user interrupt.")
    finally:
        print(
            "Placing plane back at start location.\n"
            "This allows re-running the script without restarting X-Plane"
        )
        time.sleep(5)
        exp.actor.place_at_start_position()

with skip_run("skip", "add_objects") as check, check():
    # Setup the environment and experiment
    config["experiment"]["type"] = TaxiingExperiment
    config["experiment"]["experiment_config"] = yaml.load(
        open("experiments/taxiing/experiment_config.yaml"), Loader=yaml.SafeLoader
    )
    xplane_env = NUPlaneEnv(config, debug=True)
    obs, reward, done, info = xplane_env.reset()

    # Controller or RL
    controller = PathController(agent=xplane_env.experiment.hero)

    path = "Custom Scenery/X-Plane Landmarks - New York/objects/Lady_Liberty.obj"
    path = "Custom Scenery/Simple_Ground_Equipment_and_Services_v67.8/Simple_Ground_Equipment_and_Services/Ground_carts/308_XPJavelin3.obj"
    path = "Custom Scenery/Simple_Ground_Equipment_and_Services_v67.8/Simple_Ground_Equipment_and_Services/MisterX_Lib/Stairs/Generic.obj"

    xplane_env.core.client.loadOBJ(path=path, on_ground=1)
