"""
main.py — 机器人 RPC 控制示例
只需修改 ROBOT_IP 即可连接目标机器人
"""

from rpc_client import RpcClient, send_rpcsy, send_rpc_async

# ======================================================================
#  修改此处 IP 即可
# ======================================================================
ROBOT_IP = "192.168.11.11"

# ======================================================================
#  指令定义
# ======================================================================

INIT_CMDS = [
    "{Clear}",
    "{Disable}",
    "{Mode}",
    "{SetMaxToq}",
    "{Recover}",
    "{SetRate}",
    "{Enable}",
    "{Var --clear}",
    "{Recover}",
    "{Var --type=jointtarget --name=j0 --value={0,0,0,0,0,0,0,0,0,0}}",
    "{Var --type=jointtarget --name=j1 --value={0.1,-1.5,0,0,0,0,0,0,0,0}}",
    "{Var --type=jointtarget --name=j2 --value={0.2,0,0,0,0,0,0,0,0,0}}",
    "{Var --type=jointtarget --name=j3 --value={-0.1,0,0,0,0,0,0,0,0,0}}",
    "{Var --type=jointtarget --name=j4 --value={-0.2,0,0,0,0,0,0,0,0,0}}",
]

MOTION_CMDS = [
    "{MoveAbsJ --jointtarget_var=j0}",
    "{MoveAbsJ --jointtarget_var=j1}",
]

SPEEDL_CMDS = [
    "{SpeedL --vel={0.01,0,0,0,0,0} --last_count=1000}",
    "{SpeedL --vel={-0.01,0,0,0,0,0} --last_count=1000}",
    "{SpeedL --vel={0.01,0,0,0,0,0} --last_count=1000}",
    "{SpeedL --vel={-0.01,0,0,0,0,0} --last_count=1000}",
    "{Stop}",
    "{Start}",
]

YOUR_CMDS = [
    # 在此添加自定义指令
]

# ======================================================================
#  主流程
# ======================================================================


def main():
    print(f"Connecting to {ROBOT_IP} ...")
    client = RpcClient(ROBOT_IP)

    if not client.is_connected():
        print(f"Connection failed: {client.error_info()}")
        return
    print("Connected!")

    # 1. 初始化
    send_rpcsy(client, INIT_CMDS, timeout_ms=500, sleep_s=0.1)

    # 2. 主循环
    for _ in range(3):
        # send_rpcsy(client, MOTION_CMDS, timeout_ms=50000, sleep_s=0.01)
        send_rpc_async(client, SPEEDL_CMDS, timeout_ms=10000, wait_s=0.5)


if __name__ == "__main__":
    main()
