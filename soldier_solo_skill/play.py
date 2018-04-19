from src.env.LOLEnv import LOLEnv
from src.utils.training_utils import *
from src.utils.game_utils import *
from src.models.models import mlp
from src.models.learn import DeepQNetwork
from src.constants.Defines import *
from src.constants.config import *
from src.players.defensive_ai import LOLAI

import time
import queue
import copy
import tensorflow as tf
import threading as td


def play(ip, port, self_ai, parent_ai, data_queue, score_queue):

    env = LOLEnv(ip, port)
    rule = LOLAI(CAMP_BLUE)
    explorer = LOLAI(CAMP_RED)

    obs = Observation()
    history = History(HISTORY_LENGTH)
    episode = 0

    while True:
            # ========================================== start a new episode ==========================================
            frame_counter = 0  # record the num of frames which are stored, not include the skipped ones
            synchronize_version(self_ai, parent_ai)
            obs.build(env.reset()[1])
            episode += 1
            score = 0  # accumulated reward along the whole episode

            time_start = time.time()
            reward = 0

            while True:
                # ======================================= select ai and act ===========================================
                ai_id = frame_counter % 2

                if ai_id == CAMP_BLUE:
                    ai = rule
                    obs_prev = copy.deepcopy(obs)

                    # act
                    obs.consume_frame(SKIP_FRAME, env, ai.act(obs))

                    winner = obs.judge_winner()
                    reward = obs.shape_reward(obs_prev, CAMP_RED, winner)  # always consider the red's reward

                    # store history in this section only when someone wins
                    if winner:
                        done = 1.
                        history.put((feature_prev, output, reward, feature, done))
                        if history.full():
                            data_queue.put(history.get())
                        score += reward

                else:
                    ai = self_ai
                    obs_prev = copy.deepcopy(obs)
                    feature_prev = obs.extract_feature(CAMP_RED, SOLDIERS_CONSIDER)
                    output = ai.act(feature_prev, obs, port, explorer)

                    # act
                    obs.consume_frame(SKIP_FRAME, env, Action.wrap_action(output, obs, CAMP_RED))

                    winner = obs.judge_winner()
                    reward += obs.shape_reward(obs_prev, CAMP_RED, winner)

                    done = 1. if winner is not None else 0.
                    feature = obs.extract_feature(CAMP_RED, SOLDIERS_CONSIDER)
                    history.put((feature_prev, output, reward, feature, done))
                    if history.full():
                        data_queue.put(history.get())
                    score += reward

                if winner is not None or frame_counter >= MAX_FRAMES_PER_EPISODE:
                    time_cost = time.time() - time_start
                    print()
                    print("Port {}'s episode {} ends, time : {:.2f}s, every player has {:.2f} frames in 1s."
                          .format(port, episode, time_cost, frame_counter / 2. / time_cost))

                    if winner is not None:
                        if winner == CAMP_RED:
                            print('CAMP_RED win!')
                        elif winner == CAMP_BLUE:
                            print('CAMP_BLUE win!')
                    else:
                        print('No winner...')

                    print("Score: {:2f}".format(score))
                    if port % 5 == 0:
                        score_queue.put(score)
                    print("Epsilon : %f" % self_ai.epsilon)
                    print()
                    env.end()
                    break

                frame_counter = frame_counter + 1


if __name__ == '__main__':

    Sess = tf.Session()
    global_ai = DeepQNetwork(
        n_features=QUANTITY_FEATURES,
        n_actions=QUANTITY_ACTIONS,
        scope='global_ai',
        model=mlp,
        parent_ai=None,
        sess=Sess,
        learning_rate=5e-3,
        n_replace_target=50,
        hiddens=HIDDENS,
        decay=0.99,
        memory_size=1000000,
        batch_size=3000,
        epsilon_decrement=5e-4,
        epsilon_lower=0.2,
        learn_start=LEARN_START,
    )

    dataQ = queue.Queue()

    score_plotter = ScorePlotter()
    scoreQ = queue.Queue()

    ais = []
    for i in range(N_GAMES):
        ais.append(
            DeepQNetwork(
                n_features=QUANTITY_FEATURES,
                n_actions=QUANTITY_ACTIONS,
                scope='local_ai_' + str(i),
                model=mlp,
                parent_ai=global_ai,
                sess=Sess,
                hiddens=HIDDENS
            )
        )

    Saver = tf.train.Saver()

    if RESTORE:
        Saver.restore(Sess, RESTORE_PATH)
        if RESET_EPSILON:
            Sess.run(global_ai.reset_epsilon)
        print('restored successfully from ' + RESTORE_PATH)
    else:
        Sess.run(tf.global_variables_initializer())

    for i in range(N_GAMES):
        new_thread = td.Thread(target=play, args=(IP, PORT[i], ais[i], global_ai, dataQ, scoreQ))
        new_thread.start()

    while LEARNING:
        plot_score(score_plotter, scoreQ)
        fetch_data(global_ai, dataQ)
        global_ai.learn()
        if not ONLY_PLAY and global_ai.learn_step % SAVE_EVERY == 1:
            save_path = Saver.save(Sess, SAVE_PATH + str(global_ai.learn_step) + '.ckpt')
            print('saved in' + save_path)
