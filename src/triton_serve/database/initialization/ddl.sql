-- Create database

DROP DATABASE IF EXISTS devices_db;
CREATE DATABASE devices_db;

-- public.machines definition

-- Drop table

-- DROP TABLE public.machines;

CREATE TABLE public.machines (
	host_id int4 NOT NULL DEFAULT nextval('resources_host_id_seq'::regclass),
	host_name varchar(50) NOT NULL,
	num_cpus int4 NOT NULL,
	total_memory int4 NOT NULL,
	CONSTRAINT resources_pkey PRIMARY KEY (host_id)
);


-- public.devices definition

-- Drop table

-- DROP TABLE public.devices;

CREATE TABLE public.devices (
	device_id text NOT NULL,
	host_id int4 NOT NULL,
	"name" varchar(50) NOT NULL,
	memory int4 NOT NULL,
	"index" int4 NOT NULL,
	CONSTRAINT "Devices_pkey" PRIMARY KEY (device_id),
	CONSTRAINT "FK_host_id_host_id" FOREIGN KEY (host_id) REFERENCES public.machines(host_id) ON DELETE CASCADE ON UPDATE CASCADE
);


-- public.services definition

-- Drop table

-- DROP TABLE public.services;

CREATE TABLE public.services (
	service_name varchar(50) NOT NULL,
	models _varchar NOT NULL,
	created_at int4 NULL,
	assigned_device text NULL,
	CONSTRAINT "Services_pkey" PRIMARY KEY (service_name),
	CONSTRAINT fk_deviceidassigneddevice FOREIGN KEY (assigned_device) REFERENCES public.devices(device_id)
);