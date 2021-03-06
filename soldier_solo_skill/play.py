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
import pickle
import tensorflow as tf
import threading as td


def play(ip, port, self_ai, parent_ai, data_queue, score_queue):

    episode = 0

    while True:
        try:
            env = LOLEnv(ip, port)
            rule = LOLAI(CAMP_BLUE)
            explorer = LOLAI(CAMP_RED)
            fm = FrameMonitor()

            obs = Observation()
            history = History(HISTORY_LENGTH)

            while True:
                # ========================================= start a new episode =======================================
                frame_counter = 0  # record the num of frames which are stored, not include the skipped ones
                synchronize_version(self_ai, parent_ai)
                obs.build(env.reset()[1])
                episode += 1
                time_start = time.time()

                hero_sars = SARS(history, data_queue, CAMP_RED)
                score = 0  # accumulated reward along the whole episode added from two players

                while True:
                    # ===================================== select ai and act =========================================
                    ai_camp = frame_counter % 2
                    winner = obs.judge_winner()

                    if ai_camp == CAMP_BLUE:
                        if winner is None:
                            # select rule-based AI as player and make decision
                            ai = rule
                            # take action
                            obs.consume_frame(SKIP_FRAME, env, ai.act(obs))
                        else:
                            # when the game is end, shape reward sand save trajectory for NN players
                            score += hero_sars.end(obs)

                    else:  # CAMP_RED
                        # put s-a-r-s' trajectory into history
                        score += hero_sars.end(obs)

                        if winner is None:
                            # select NN as player and make decision
                            ai = self_ai
                            # hero_sars.s1 is now the newest extracted feature
                            output = ai.act(hero_sars.s1, obs, port, explorer)
                            # save action
                            hero_sars.get_action(output)
                            # take action
                            obs.consume_frame(SKIP_FRAME, env, Action.wrap_action(output, obs, CAMP_RED))

                    # ===================================== when the game ends ========================================
                    if winner is not None or frame_counter >= MAX_FRAMES_PER_EPISODE:
                        print()
                        # ============= time printer =============
                        time_cost = time.time() - time_start
                        player_frames = frame_counter / (PLAYERS_EVERY_SIDE * 2) / time_cost
                        print("Port {}'s episode {} ends, time : {:.2f}s, every player has {:.2f} frames in 1s."
                              .format(port, episode, time_cost, player_frames))
                        # ============ winner printer ============
                        if winner is not None:
                            if winner == CAMP_RED:
                                print('CAMP_RED win!')
                            elif winner == CAMP_BLUE:
                                print('CAMP_BLUE win!')
                        else:
                            print('No winner...')
                        # ============ score handler =============
                        print("Score: {:2f}".format(score))
                        if port % 5 == 0:
                            score_queue.put(score)
                        print("Epsilon : %f" % self_ai.epsilon)
                        print()
                        env.end()
                        # ============ frame monitor =============
                        if fm.few_frames(player_frames):
                            raise Exception("few frames")

                        break

                    # ======================================== count frames ===========================================
                    frame_counter += 1

        except Exception as e:
            # when the client closed or too few frames, restart the socket
            if e.args == ("unpack requires a bytes object of length 4",) or e.args == ("few frames",):
                print("client on port", port, "closed, wait for reconnection...")
                env.close()
                time.sleep(30)


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
        memory_size=3000000,
        batch_size=3000,
        epsilon_decrement=1e-4,
        epsilon_lower=0.2,
        learn_start=30000,
        double=True,
        dueling=True
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
            # with open(SAVE_PATH + str(global_ai.learn_step), 'wb') as f:
            #     pickle.dump(np.array(evaluation_set), f, pickle.HIGHEST_PROTOCOL)
            print('saved in' + save_path)
