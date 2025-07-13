#!/bin/bash

# 智能运维 LangGraph 服务器启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_message() {
    echo -e "${2}${1}${NC}"
}

print_message "🚀 智能运维 LangGraph 服务器启动脚本" "$BLUE"
echo

# 检查虚拟环境
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_message "❌ 虚拟环境未激活，正在激活..." "$YELLOW"
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        print_message "✅ 虚拟环境已激活" "$GREEN"
    else
        print_message "❌ 虚拟环境不存在，请先运行 'uv sync' 创建虚拟环境" "$RED"
        exit 1
    fi
else
    print_message "✅ 虚拟环境已激活: $VIRTUAL_ENV" "$GREEN"
fi

# 检查配置文件
if [ ! -f ".env" ]; then
    print_message "❌ .env 文件不存在，正在创建..." "$YELLOW"
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_message "✅ .env 文件已创建，请编辑配置您的 API 密钥" "$GREEN"
        print_message "📝 请编辑 .env 文件并配置 DEEPSEEK_API_KEY" "$YELLOW"
    else
        print_message "❌ .env.example 文件不存在" "$RED"
        exit 1
    fi
fi

# 检查 langgraph.json
if [ ! -f "langgraph.json" ]; then
    print_message "❌ langgraph.json 配置文件不存在" "$RED"
    exit 1
fi

# 检查依赖
print_message "📦 检查依赖..." "$BLUE"
if ! command -v langgraph &> /dev/null; then
    print_message "❌ langgraph-cli 未安装，正在安装..." "$YELLOW"
    uv add "langgraph-cli[inmem]"
    print_message "✅ langgraph-cli 安装完成" "$GREEN"
fi

# 安装项目
print_message "📦 安装项目..." "$BLUE"
uv pip install -e .
print_message "✅ 项目安装完成" "$GREEN"

# 解析命令行参数
PORT=2024
HOST="127.0.0.1"  # 改为 localhost，更安全且可访问
NO_BROWSER=true
SERVER_LOG_LEVEL="INFO"

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --browser)
            NO_BROWSER=false
            shift
            ;;
        --debug)
            SERVER_LOG_LEVEL="DEBUG"
            shift
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo "选项:"
            echo "  --port PORT        设置服务器端口 (默认: 2024)"
            echo "  --host HOST        设置服务器主机 (默认: 127.0.0.1)"
            echo "                     使用 0.0.0.0 允许外部访问"
            echo "  --browser          启动时打开浏览器"
            echo "  --debug            启用调试模式"
            echo "  --help             显示此帮助信息"
            exit 0
            ;;
        *)
            print_message "❌ 未知选项: $1" "$RED"
            exit 1
            ;;
    esac
done

# 显示启动信息
print_message "🔧 服务器配置:" "$BLUE"
echo "  - 端口: $PORT"
echo "  - 主机: $HOST"
echo "  - 浏览器: $([ "$NO_BROWSER" = true ] && echo "不启动" || echo "启动")"
echo "  - 日志级别: $SERVER_LOG_LEVEL"
echo

# 构建启动命令
CMD="$VIRTUAL_ENV/bin/langgraph dev --port $PORT --host $HOST --server-log-level $SERVER_LOG_LEVEL"
if [ "$NO_BROWSER" = true ]; then
    CMD="$CMD --no-browser"
fi

print_message "🚀 启动 LangGraph 服务器..." "$GREEN"
echo "命令: $CMD"
echo

# 显示访问地址
print_message "📍 服务器地址:" "$BLUE"
# 如果 HOST 是 0.0.0.0，显示 localhost 作为访问地址
if [ "$HOST" = "0.0.0.0" ]; then
    ACCESS_HOST="127.0.0.1"
else
    ACCESS_HOST="$HOST"
fi
echo "  - 🚀 API: http://$ACCESS_HOST:$PORT"
echo "  - 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://$ACCESS_HOST:$PORT"
echo "  - 📚 API Docs: http://$ACCESS_HOST:$PORT/docs"
echo

# 显示停止提示
print_message "💡 提示: 按 Ctrl+C 停止服务器" "$YELLOW"
echo

# 启动服务器
exec $CMD