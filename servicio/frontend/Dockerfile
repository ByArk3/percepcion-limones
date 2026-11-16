# Frontend: página web estática servida con nginx (sin frameworks ni Node).
FROM nginx:1.28.0-alpine
COPY servicio/frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY servicio/frontend/web/ /usr/share/nginx/html/
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=5s --retries=3 CMD wget -q -O /dev/null http://localhost/ || exit 1
